#!/usr/bin/env python3
"""Train Go2 controller policy in MuJoCo using stable-baselines3 PPO.

This script aims to reproduce the training style of the original
`legged_gym/scripts/train.py` call (PPO, many environments, seeds,
logging). It trains a policy that maps controller observations to
controller actions (num_actions) and saves the trained model.
"""
import os
import sys
import argparse
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import yaml
import numpy as np
# from stable_baselines3 import PPO
from training.utils.ppo import PPO
import types
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.logger import configure
# from stable_baselines3.common.buffers import RolloutBuffer
from training.utils.buffers import RolloutBuffer
from training.utils.train_env import TrainEnv
from training.utils.test_env import TestEnv, TerrainGymEnv

# Silence Gym -> Gymnasium migration warning and noisy gym/gymnasium loggers
import warnings
import logging
warnings.filterwarnings("ignore", message=".*Please upgrade to Gymnasium.*", category=UserWarning)
logging.getLogger("gym").setLevel(logging.ERROR)
logging.getLogger("gymnasium").setLevel(logging.ERROR)


def make_env_fn(normal_policy, max_episode_steps=1000, nade=False, criticality_model=None, critical_threshold=0.5):
    def _thunk():
        # Create a lightweight dummy policy to satisfy TestEnv/Go2Controller
        config_file_path = "go2_training.yaml"
        terrain_cfg = "terrain_config.yaml"
        trainer = TestEnv(policy=normal_policy, config_file_path=config_file_path, terrain_config_file=terrain_cfg, critical_threshold=critical_threshold)
        env = TrainEnv(trainer=trainer, max_episode_steps=max_episode_steps, nade=nade, criticality_model=criticality_model)
        return Monitor(env)

    return _thunk

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rl_device', type=str, default='cuda:2')
    parser.add_argument('--max_steps', type=int, default=30)
    parser.add_argument('--n_eval_episodes', type=int, default=5, help='Number of episodes per evaluation')
    parser.add_argument('--out', type=str, default='training/models/')
    parser.add_argument('--run_name', type=str, default='run_offline1_new', help='Subdirectory name for this training run (default: run_offline2_2)')
    parser.add_argument('--pretrain', type=str, default='training/models/actor_init.zip',
                        help='Path to a pretrained PyTorch model or SB3 .zip to initialize normal policy (default uses training/models/actor_init.zip)')
    parser.add_argument('--criticality_model_path', type=str, default='criticality/stage1_plus/model/stage1_plus_criticality_best_new_3.pt', help='Path to criticality model')
    # parser.add_argument('--initial', type=str, default='training/models/actor_init.zip')
    parser.add_argument('--initial', type=str, default='training/models/run_offline1_new/run_offline1_new_offline_ep50.policy.pt')
    parser.add_argument('--dataset', type=str, default='/mnt/mnt1/linxuan/go2_data/data/training/round1_new', help='Path to offline dataset directory')
    parser.add_argument('--offline_epochs', type=int, default=50, help='Epochs for offline training')
    parser.add_argument('--offline_batch_size', type=int, default=512)
    parser.add_argument('--offline_lr', type=float, default=1e-3)
    parser.add_argument('--value_coef', type=float, default=1.0, help='Weight for value regression loss during offline training')
    args = parser.parse_args()
    args.out = os.path.join(args.out, args.run_name)
    print(args)

    os.makedirs(args.out, exist_ok=True)

    # If a pretrained SB3 .zip is provided, load it only for weight extraction (no env attachment)
    # We will always create a fresh PPO model below and copy weights into it to avoid carrying over
    # stale internal training state (rollout buffer, optim state, etc.) from the checkpoint.
    sb3_pretrain_model = None
    if args.pretrain.endswith('.zip'):
        from stable_baselines3 import PPO as SB3PPO
        print('Found SB3 pretrained .zip at', args.pretrain, '- will map its policy into a fresh model')
        sb3_pretrain_model = SB3PPO.load(args.pretrain, device='cpu')

    class PolicyOnlyWrapper(torch.nn.Module):
        def __init__(self, net_pi, act_net):
            super().__init__()
            self.net_pi = net_pi
            self.act_net = act_net

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            latent = self.net_pi(x)
            actions = self.act_net(latent)
            return actions
        
    # create vec env
    # create envs with rendering disabled by default
    pretrain_wrapper = PolicyOnlyWrapper(sb3_pretrain_model.policy.mlp_extractor.policy_net.to('cpu').eval(), sb3_pretrain_model.policy.action_net.to('cpu').eval()).cpu()

    # load criticality model
    from criticality.utils.criticality_model import SimpleClassifier
    criticality_model = SimpleClassifier(input_dim=56)
    criticality_model.load_state_dict(torch.load(args.criticality_model_path, map_location='cpu'))
    criticality_model.to('cpu').eval()
    
    env_fns = [make_env_fn(pretrain_wrapper)]
    vec_env = DummyVecEnv(env_fns)

    # Create a fresh model and try to copy compatible weights if available
    hidden_sizes = [512, 256, 128]
    policy_kwargs = {'net_arch': dict(pi=hidden_sizes, vf=hidden_sizes), 'activation_fn': torch.nn.ELU}
    # Create model with CLI-configured hyperparameters
    model = PPO(
        'MlpPolicy', vec_env, verbose=1, device=args.rl_device,
        policy_kwargs=policy_kwargs,
    )

    if args.initial:
        # prefer already-loaded sb3_pretrain_model if available
        if args.initial.endswith('.zip'):
            initial_model = sb3_pretrain_model if sb3_pretrain_model is not None else __import__('stable_baselines3').PPO.load(args.initial, device='cpu')
            src_sd = initial_model.policy.state_dict()
        elif args.initial.endswith('.pt'):
            state_dict = torch.load(args.initial, map_location='cpu')
            src_sd = state_dict['policy_state_dict']

        # Source and destination state dicts
        policy_sd = model.policy.state_dict()

        matched = {}
        used_src = set()

        # 1) Exact name + shape matches
        for src_k, src_v in src_sd.items():
            if src_k in policy_sd and tuple(policy_sd[src_k].shape) == tuple(src_v.shape):
                policy_sd[src_k].copy_(src_v)
                matched[src_k] = src_k
                used_src.add(src_k)

        # 2) Suffix (last token) + shape matches for remaining dst keys
        for dst_k in list(policy_sd.keys()):
            if dst_k in matched:
                continue
            dst_shape = tuple(policy_sd[dst_k].shape)
            dst_suffix = dst_k.split('.')[-1]
            for src_k, src_v in src_sd.items():
                if src_k in used_src:
                    continue
                if src_k.split('.')[-1] == dst_suffix and tuple(src_v.shape) == dst_shape:
                    policy_sd[dst_k].copy_(src_v)
                    matched[dst_k] = src_k
                    used_src.add(src_k)
                    break

        # 3) Shape-only matching for any remaining dst keys (first-fit)
        for dst_k in list(policy_sd.keys()):
            if dst_k in matched:
                continue
            dst_shape = tuple(policy_sd[dst_k].shape)
            for src_k, src_v in src_sd.items():
                if src_k in used_src:
                    continue
                if tuple(src_v.shape) == dst_shape:
                    policy_sd[dst_k].copy_(src_v)
                    matched[dst_k] = src_k
                    used_src.add(src_k)
                    break

        model.policy.load_state_dict(policy_sd)
        print(f'Initialized policy from {args.initial} — matched {len(matched)} tensors')

    def load_offline_dataset(data_dir):
        all_obs, all_acts, all_returns, all_weights = np.zeros((0, 48)), np.zeros((0, 12)), np.zeros((0,)), np.zeros((0,))
        gamma = getattr(model, 'gamma', 0.99)
        for filename in os.listdir(data_dir):
            if filename.endswith('.npy'):
                path = os.path.join(data_dir, filename)
                data = np.load(path, allow_pickle=True).item()
                obs = np.array(data['obs'], dtype=np.float32)
                acts = np.array(data['actions'], dtype=np.float32)
                rews = np.array(data['rewards'], dtype=np.float32)
                dones = np.array(data['dones'], dtype=np.float32)
                useful = np.array(data['useful'], dtype=bool)
                weights = np.array(data['weights'], dtype=np.float32)

                # compute discounted returns per episode
                returns = np.zeros_like(rews, dtype=np.float32)
                G = 0.0
                for i in reversed(range(len(rews))):
                    if dones[i]:
                        G = rews[i]
                    else:
                        G = rews[i] + gamma * G
                    returns[i] = G

                useful_idx = np.where(useful)[0]

                all_obs = np.concatenate([all_obs, obs[useful_idx]])
                all_acts = np.concatenate([all_acts, acts[useful_idx]])
                all_returns = np.concatenate([all_returns, returns[useful_idx]])
                all_weights = np.concatenate([all_weights, weights[useful_idx]])

        obs_t = torch.tensor(all_obs, dtype=torch.float32)
        acts_t = torch.tensor(all_acts, dtype=torch.float32)
        returns_t = torch.tensor(all_returns, dtype=torch.float32)
        weights_t = torch.tensor(all_weights, dtype=torch.float32)
        return obs_t, acts_t, returns_t, weights_t

    def offline_train_policy(model, dataset_path, epochs, batch_size, lr, value_coef, device='cpu'):
        from torch.utils.data import TensorDataset, DataLoader
        obs_t, acts_t, returns_t, weights_t = load_offline_dataset(dataset_path)
        ds = TensorDataset(obs_t, acts_t, returns_t)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

        policy = model.policy
        # Prefer to use mlp_extractor.policy_net/value_net and action_net/value_net heads
        params = [p for p in policy.parameters() if p.requires_grad]
        optim = torch.optim.Adam(params, lr=lr)

        for ep in range(1, epochs+1):
            epoch_loss = 0.0
            for b_obs, b_act, b_ret in dl:
                b_obs = b_obs.to(device)
                b_act = b_act.to(device)
                b_ret = b_ret.to(device)

                # forward through policy pi head
                latent_pi = policy.mlp_extractor.policy_net(b_obs)
                pred_act = policy.action_net(latent_pi)
                pred_act = torch.clip(pred_act, -4, 4)

                # value prediction
                latent_v = policy.mlp_extractor.value_net(b_obs)
                pred_val = policy.value_net(latent_v).squeeze(-1)
                # import pdb; pdb.set_trace()

                # value loss
                val_loss = F.mse_loss(pred_val, b_ret)

                # advantage (returns - value)
                adv = (b_ret - pred_val.detach())
                adv_weights = torch.clamp(adv, min=0.0)
                if adv_weights.sum() > 0:
                    adv_weights = adv_weights / (adv_weights.mean() + 1e-8)

                # policy dist: treat as regression to actions (MSE), weight by advantages
                per_sample_mse = ((pred_act - b_act)**2).mean(dim=1)
                policy_loss = (adv_weights * per_sample_mse).mean() if adv_weights.sum() > 0 else per_sample_mse.mean()

                loss = policy_loss + value_coef * val_loss

                optim.zero_grad()
                loss.backward()
                optim.step()

                epoch_loss += float(loss.item())

            avg = epoch_loss / (len(dl) if len(dl) > 0 else 1)
            print(f'[Offline] epoch={ep}/{epochs} loss={avg:.6f}')

            # save final offline-updated policy
            try:
                save_to = os.path.join(args.out, f'{args.run_name}_offline_ep{ep}.policy.pt')
                safe_model_save(model, save_to, verbose=1)
            except Exception as e:
                print('[Offline] saving failed:', e)


    def evaluate_policy(safe_policy, policy=pretrain_wrapper, criticality_model=criticality_model, n_episodes: int = 5):
        config_file_path = "go2_training.yaml"
        terrain_cfg = "terrain_config.yaml"

        policy_net = safe_policy.mlp_extractor.policy_net
        action_net = safe_policy.action_net
        policy_net.to('cpu')
        action_net.to('cpu')
        policy_net.eval()
        action_net.eval()

        safe_wrapper = PolicyOnlyWrapper(policy_net, action_net).cpu()

        trainer = TestEnv(policy=policy, safe_policy=safe_wrapper, criticality_model=criticality_model, config_file_path=config_file_path, terrain_config_file=terrain_cfg, critical_threshold=0.5)
        env = TerrainGymEnv(trainer, max_episode_steps=args.max_steps)

        action_space = env.action_space
        # prepare discretization edges for actions: 10 bins per dimension
        low = np.asarray(action_space.low, dtype=np.float32)
        high = np.asarray(action_space.high, dtype=np.float32)
        high = np.where(high == low, low + 1.0, high)
        action_edges = [np.linspace(low[d], high[d], num=11) for d in range(low.shape[0])]

        results = []
        for i in range(n_episodes):
            obs, _ = env.reset()
            done = False

            # per-episode accumulators
            ep_steps = 0
            ep_total_reward = 0.0

            while not done:
                ep_steps += 1
                bins = np.random.randint(0, 10, size=action_space.shape)
                centers = np.zeros(action_space.shape, dtype=np.float32)
                flat_bins = np.asarray(bins).reshape(-1)
                for d in range(flat_bins.shape[0]):
                    b = int(flat_bins[d])
                    e = action_edges[d]
                    centers.reshape(-1)[d] = 0.5 * (e[b] + e[b+1])
                action = centers
                action = np.asarray(action, dtype=np.float32)

                next_obs, reward, terminated, truncated, info = env.step(action)
                obs = next_obs

                ep_total_reward += reward

                done = bool(terminated) or bool(truncated) or bool(info.get('fallen', False) or info.get('collided', False) or info.get('base_collision', False) or info.get('thigh_collision', False) or info.get('stuck', False))

            # episode finished; determine crash/failure from last step's info
            crash = int(bool(info.get('fallen', False) or info.get('collided', False) or info.get('base_collision', False) or info.get('thigh_collision', False) or info.get('stuck', False)))
            crash_type = 'fallen' if info.get('fallen', False) else 'collided' if info.get('collided', False) else 'base_collision' if info.get('base_collision', False) else 'thigh_collision' if info.get('thigh_collision', False) else 'stuck' if info.get('stuck', False) else 'safe'

            # print per-episode breakdown and total
            print(f"episode {i+1}/{n_episodes} steps={ep_steps} crash={crash}, type={crash_type}")
            print(f"  total_reward (accumulated) = {ep_total_reward:.6f}")
            results.append(ep_total_reward)

        return results

    def safe_model_save(model_obj, save_path, verbose=1):
        """Save `model_obj` while temporarily clearing attributes that may be
        unpickleable (eg. multiprocessing auth keys inside VecEnv/processes).
        Restores cleared attributes after the save attempt.
        """
        cleared = {}
        for attr in ('env', 'envs', 'training_env', 'vec_env'):
            if hasattr(model_obj, attr):
                cleared[attr] = getattr(model_obj, attr)
                try:
                    setattr(model_obj, attr, None)
                except Exception:
                    # best-effort: if we can't clear it, ignore and continue
                    cleared.pop(attr, None)

        try:
            # try to save only the policy state dict with torch
            fb_dir = os.path.dirname(save_path)
            if fb_dir:
                os.makedirs(fb_dir, exist_ok=True)
            fb_path = save_path
            # prefer .pt for fallback
            if not fb_path.endswith('.pt'):
                fb_path = save_path + '.pt'
            torch.save({'policy_state_dict': model_obj.policy.state_dict()}, fb_path)
            if verbose:
                print(f'[safe_model_save] saved policy state_dict to {fb_path}')
            return True
        finally:
            for attr, val in cleared.items():
                try:
                    setattr(model_obj, attr, val)
                except Exception:
                    pass

    print('Starting offline PPO-style training (adv-weighted BC + value regression)')
    offline_train_policy(model, args.dataset, args.offline_epochs, args.offline_batch_size, args.offline_lr, args.value_coef, device=args.rl_device)
    # evaluate after offline training
    _ = evaluate_policy(model.policy, n_episodes=int(args.n_eval_episodes))

    save_path = os.path.join(args.out, f'{args.run_name}_ppo.zip')
    if safe_model_save(model, save_path, verbose=1):
        print('Saved model to', save_path)
    else:
        print('Failed to save final model to', save_path)

if __name__ == '__main__':
    main()
