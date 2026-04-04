"""
Stage2: simple RL fine-tuning scaffold.

This script loads a pretrained stage1 classifier and uses it as a reward-shaping penalty
during RL training of a terrain policy. It uses stable-baselines3 PPO if available.

Note: this is a scaffold — adapt hyperparams and policy architecture to your setup.
"""
import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
except Exception:
    PPO = None

import sys, os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from deploy_mujoco.terrain_trainer import TerrainTrainer, TerrainGymEnv
from criticality.utils.criticality_model import SimpleClassifier

# try to import MyLander's DQN utilities
MYLANDER_UTILS = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'MyLander', 'Env_agent', 'utils'))
# dynamically load dqn.py from MyLander utils if present (avoid static import errors)
DQN = None
ReplayBuffer = None
if os.path.exists(os.path.join(MYLANDER_UTILS, 'dqn.py')):
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location('dqn', os.path.join(MYLANDER_UTILS, 'dqn.py'))
        dqn_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dqn_mod)
        DQN = getattr(dqn_mod, 'DQN', None)
        ReplayBuffer = getattr(dqn_mod, 'ReplayBuffer', None)
    except Exception:
        DQN = None
        ReplayBuffer = None


class ShapedEnv:
    """Wrapper around TerrainGymEnv that subtracts classifier score as penalty."""
    def __init__(self, env, classifier, device='cpu', penalty_scale=1.0):
        self.env = env
        self.classifier = classifier.to(device)
        self.device = device
        self.penalty_scale = penalty_scale
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self):
        obs, info = self.env.reset()
        return obs

    def step(self, action):
        obs_next, reward, terminated, truncated, info = self.env.step(action)
        # classifier expects single-step obs (np array)
        with torch.no_grad():
            x = torch.from_numpy(obs_next.astype(np.float32)).unsqueeze(0).to(self.device)
            logits = self.classifier(x)
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().item()
        shaped_reward = reward - self.penalty_scale * float(probs)
        return obs_next, shaped_reward, terminated, truncated, info


def main(args):
    # load classifier
    device = torch.device('cuda' if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu')
    # instantiate trainer/env
    go2_cfg = ("terrain", "go2.yaml")
    terrain_cfg = "terrain_config.yaml"
    trainer = TerrainTrainer(go2_cfg, terrain_cfg)
    env = TerrainGymEnv(trainer, max_episode_steps=args.max_steps)

    # load classifier model
    data_sample = trainer.get_terrain_observation()
    input_dim = data_sample.shape[0]
    classifier = SimpleClassifier(input_dim=input_dim, hidden=args.hidden)
    if args.stage1_model and os.path.exists(args.stage1_model):
        classifier.load_state_dict(torch.load(args.stage1_model, map_location='cpu'))
    classifier.eval()

    wrapped = ShapedEnv(env, classifier, device=device, penalty_scale=args.penalty_scale)

    # Use offline DQN training (adapted from MyLander stage4/3)
    if DQN is None:
        print('MyLander DQN implementation not found. Ensure Embodied/MyLander/Env_agent/utils is present.')
        return

    # load replay buffer produced by stage1 processing
    replay_path = os.path.join(args.data_dir, 'replay_buffer.npy')
    if not os.path.exists(replay_path):
        print('Replay buffer not found at', replay_path)
        return
    replay = np.load(replay_path, allow_pickle=True).tolist()

    # create replay pools: for balance we split transitions into pos/neg by reward
    pos_pool = [t for t in replay if t.get('reward', 0.0) > 0.5]
    neg_pool = [t for t in replay if t.get('reward', 0.0) <= 0.5]
    print(f'Loaded replay transitions total={len(replay)} pos={len(pos_pool)} neg={len(neg_pool)}')

    # prepare DQN agent; initial model seed from stage1 classifier
    sample_obs = trainer.get_terrain_observation()
    input_dim = sample_obs.shape[0]
    initial_model = SimpleClassifier(input_dim=input_dim, hidden=args.hidden)
    # adapt classifier output to scalar Q-value: wrap in nn.Sequential
    q_net = nn.Sequential(initial_model, nn.Linear(2,1))
    # if stage1 model exists, load weights into initial_model where compatible
    if args.stage1_model and os.path.exists(args.stage1_model):
        try:
            state = torch.load(args.stage1_model, map_location='cpu')
            # try loading into initial_model
            initial_model.load_state_dict({k.replace('net.', ''): v for k,v in state.items() if k in initial_model.state_dict()})
        except Exception:
            pass

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = torch.optim.Adam(q_net.parameters(), lr=args.lr)
    agent = DQN(q_net, optimizer, learning_rate=args.lr, gamma=args.gamma, target_update=args.target_update, device=device)

    def balanced_sample(pbs=256, nbs=1024):
        pos_samples = []
        neg_samples = []
        if len(pos_pool) > 0:
            pos_samples = list(np.random.choice(pos_pool, size=min(len(pos_pool), pbs), replace=len(pos_pool) < pbs))
        if len(neg_pool) > 0:
            neg_samples = list(np.random.choice(neg_pool, size=min(len(neg_pool), nbs), replace=len(neg_pool) < nbs))
        transitions = pos_samples + neg_samples
        state, action, reward, next_state, done = [], [], [], [], []
        for it in transitions:
            state.append(torch.tensor(it['input'], dtype=torch.float))
            action.append(it.get('action', np.array([])))
            reward.append(float(it.get('reward', 0.0)))
            next_state.append(torch.tensor(it['next_input'], dtype=torch.float))
            done.append(float(it.get('done', 0.0)))

        transition_dict = {
            'inputs': torch.stack(state, dim=0),
            'actions': action,
            'rewards': reward,
            'next_inputs': torch.stack(next_state, dim=0),
            'dones': done
        }
        return transition_dict

    # training loop
    os.makedirs(args.save_dir, exist_ok=True)
    best_auc = 0.0
    for epoch in range(1, args.epochs + 1):
        trans = balanced_sample(pbs=args.pos_bs, nbs=args.neg_bs)
        agent.update(trans)
        if epoch % args.target_update == 0:
            agent.target_q_net.load_state_dict(agent.q_net.state_dict())
        if epoch % args.validate_interval == 0:
            # simple validation: compute average q on pos/neg val slices
            # load pos/neg validation if available
            val_pos = None
            val_neg = None
            try:
                val_pos = np.load(os.path.join(args.data_dir, 'pos.npy'))
                val_neg = np.load(os.path.join(args.data_dir, 'neg.npy'))
            except Exception:
                pass
            if val_pos is not None and val_neg is not None:
                q_vals = []
                labels = []
                with torch.no_grad():
                    for arr, lab in [(val_pos,1),(val_neg,0)]:
                        for i in range(min(len(arr), args.val_samples_per_class)):
                            x = torch.from_numpy(arr[i].astype(np.float32)).unsqueeze(0).to(device)
                            out = agent.q_net(x).cpu().numpy().item()
                            q_vals.append(out)
                            labels.append(lab)
                # compute simple AUC-like via PR? use thresholding
                q_vals = np.array(q_vals)
                labels = np.array(labels)
                # compute simple separability metric: mean pos > mean neg
                mean_pos = q_vals[labels==1].mean() if (labels==1).any() else 0.0
                mean_neg = q_vals[labels==0].mean() if (labels==0).any() else 0.0
                print(f'val mean_pos={mean_pos:.4f} mean_neg={mean_neg:.4f}')
                torch.save(agent.q_net.state_dict(), os.path.join(args.save_dir, f'dqn_epoch{epoch}.pt'))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage1_model', default='stage1/model/stage1_classifier_best.pt')
    parser.add_argument('--save_dir', default='stage2/model')
    parser.add_argument('--data_dir', default='data/processed')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--hidden', default=256, type=int)
    parser.add_argument('--penalty_scale', default=1.0, type=float)
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--pos_bs', default=256, type=int)
    parser.add_argument('--neg_bs', default=1024, type=int)
    parser.add_argument('--lr', default=3e-5, type=float)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--target_update', default=20, type=int)
    parser.add_argument('--validate_interval', default=50, type=int)
    parser.add_argument('--val_samples_per_class', default=256, type=int)
    parser.add_argument('--max_steps', default=500, type=int)
    args = parser.parse_args()
    main(args)
