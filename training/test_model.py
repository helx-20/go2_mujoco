#!/usr/bin/env python3
"""
Simple NDE-style test for go2_mujoco: run many deterministic episodes without terrain modifications
and compute running failure statistics (mean, relative half-width).
"""
import argparse
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import numpy as np
import math
from scipy.stats import norm

from training.utils.test_env import TestEnv, TerrainGymEnv
from training.utils.train_env import TrainEnv
from stable_baselines3.common.monitor import Monitor
import torch
from stable_baselines3 import PPO


alpha = 0.05
z = norm.isf(q=alpha)


def calculate_val(the_list):
    Mean = []
    Relative_half_width = []
    Var = []
    var_old = 0
    mean_old = 0
    for i in range(len(the_list)):
        if math.isnan(the_list[i]) or math.isinf(the_list[i]):
            the_list[i] = 0.0
        n = i + 1
        mean_new = mean_old + (the_list[i] - mean_old) / n
        Mean.append(mean_new)
        var_new = (n - 1) * var_old / n + (n - 1) * (the_list[i] - mean_old) ** 2 / (n * n)
        Var.append(1.96 * (np.sqrt(var_new / n)))
        Relative_half_width.append(z * (np.sqrt(var_new / n) / (mean_new + 1e-30)))
        var_old = var_new
        mean_old = mean_new
    return Mean, Relative_half_width, Var

def make_env_fn(trainer, max_episode_steps=1000):
    def _thunk():
        env = TrainEnv(trainer=trainer, max_episode_steps=max_episode_steps)
        return Monitor(env)

    return _thunk

class PolicyOnlyWrapper(torch.nn.Module):
    def __init__(self, net_pi, act_net):
        super().__init__()
        self.net_pi = net_pi
        self.act_net = act_net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.net_pi(x)
        actions = self.act_net(latent)
        return actions

def run(args):
    # go2_config_file: (foldername, cfg filename) relative to deploy_mujoco
    config_file_path = "go2_training.yaml"
    terrain_cfg = "terrain_config.yaml"

    # If controller_path is a SB3 .zip, convert it to a TorchScript policy expected by Go2Controller
    controller_path = getattr(args, 'controller_path', None)
     # load SB3 model
    if controller_path.endswith('.zip'):
        sb3 = PPO.load(controller_path, device='cpu')
    elif controller_path.endswith('.pt'):
        hidden_sizes = [512, 256, 128]
        policy_kwargs = {'net_arch': dict(pi=hidden_sizes, vf=hidden_sizes), 'activation_fn': torch.nn.ELU}
        # Create model with CLI-configured hyperparameters
        trainer = TestEnv(policy=None, config_file_path=config_file_path, terrain_config_file=terrain_cfg)
        dummy_env = make_env_fn(trainer, max_episode_steps=1000)()  # create a dummy env to infer action space
        sb3 = PPO('MlpPolicy', dummy_env, policy_kwargs=policy_kwargs, device='cpu')
        # Load state dict from .pt file
        state_dict = torch.load(controller_path, map_location='cpu')
        state_dict = state_dict['policy_state_dict']
        sb3.policy.load_state_dict(state_dict)

    # Export a lightweight wrapper that runs policy_net -> action_net so
    # the traced module returns actions with correct shape. This omits
    # the value_net and other modules.
    policy_net = sb3.policy.mlp_extractor.policy_net
    action_net = sb3.policy.action_net
    policy_net.to('cpu')
    action_net.to('cpu')
    policy_net.eval()
    action_net.eval()

    safe_wrapper = PolicyOnlyWrapper(policy_net, action_net).cpu()
    # example = torch.zeros((1, 48), dtype=torch.float32)
    # try:
    #     traced = torch.jit.trace(wrapper, example, check_trace=False)
    #     out_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model', 'go2_controller_new.pt')
    #     os.makedirs(os.path.dirname(out_path), exist_ok=True)
    #     traced.save(out_path)
    #     print('Converted policy_net+action_net from', controller_path, '->', out_path)
    # except Exception as e:
    #     print('Failed to trace policy wrapper:', e)

    from criticality.utils.criticality_model import SimpleClassifier
    criticality_model = SimpleClassifier(input_dim=56)
    criticality_model.load_state_dict(torch.load('criticality/stage1_plus/model/stage1_plus_criticality_best_new_3.pt', map_location='cpu'))
    criticality_model.to('cpu').eval()

    from stable_baselines3 import PPO as SB3PPO
    sb3_pretrain_model = SB3PPO.load('training/models/actor_init.zip', device='cpu')
    pretrain_wrapper = PolicyOnlyWrapper(sb3_pretrain_model.policy.mlp_extractor.policy_net.to('cpu').eval(), sb3_pretrain_model.policy.action_net.to('cpu').eval()).cpu()

    training_out = getattr(args, 'training_out', None)
    if training_out is not None:
        collect_training_data = True
    else:
        collect_training_data = False
    trainer = TestEnv(policy=pretrain_wrapper, safe_policy=safe_wrapper, config_file_path=config_file_path, terrain_config_file=terrain_cfg, criticality_model=criticality_model, critical_threshold=args.critical_threshold, collect_training_data=collect_training_data)
    env = TerrainGymEnv(trainer, max_episode_steps=args.max_steps)

    n = args.episodes
    crashes = []
    out_path = args.out

    action_space = env.action_space
    # prepare discretization edges for actions: 10 bins per dimension
    low = np.asarray(action_space.low, dtype=np.float32)
    high = np.asarray(action_space.high, dtype=np.float32)
    action_edges = [np.linspace(low[d], high[d], num=11) for d in range(low.shape[0])]
    D = 4
    grids = np.meshgrid(*[np.arange(10) for _ in range(D)], indexing='ij')
    bins_flat = np.stack([g.reshape(-1) for g in grids], axis=1).astype(np.int64)
    num_actions = bins_flat.shape[0]
    centers = np.zeros((num_actions, D), dtype=np.float32)
    for d in range(D):
        e = action_edges[d]
        b_idx = bins_flat[:, d]
        centers[:, d] = 0.5 * (e[b_idx] + e[b_idx + 1])
    candidates_arr = centers

    # global buffers for criticality training data (across episodes)
    training_data_all = {'obs': [], 'actions': [], 'rewards': [], 'dones': [], 'useful': [], 'weights': []}

    for i in range(n):
        obs, _ = env.reset()
        done = False
        steps = 0
        total_weight = 1.0

        while not done:
            steps += 1
            with torch.no_grad():
                t_obs = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).repeat(candidates_arr.shape[0], 1)
                t_act = torch.from_numpy(np.asarray(candidates_arr, dtype=np.float32))
                t_in = torch.cat([t_obs, t_act], dim=1)
                t_out = criticality_model(t_in)
                criticality = torch.nn.functional.softmax(t_out, dim=1)[:, 1].squeeze().cpu().numpy()
            if not args.nade:
                idx = np.random.randint(0, candidates_arr.shape[0])
                weight = 1.0
            else:
                if np.max(criticality) > 3e-1 and total_weight > 1e-4:
                    idx = int(np.argmax(criticality))
                    weight = float((1 / len(criticality)) / (criticality[idx] / np.sum(criticality)))
                else:
                    idx = np.random.randint(0, candidates_arr.shape[0])
                    weight = 1.0
            total_weight *= weight
            action = candidates_arr[idx]
            action = np.asarray(action, dtype=np.float32)
            next_obs, reward, terminated, truncated, info = env.step(action)
            if training_out is not None:
                while len(env.trainer.training_data['weights']) < len(env.trainer.training_data['obs']):
                    env.trainer.training_data['weights'].append(weight)
            obs = next_obs
            done = bool(terminated) or bool(truncated) or bool(info.get('fallen', False) or info.get('collided', False) or info.get('base_collision', False) or info.get('thigh_collision', False) or info.get('stuck', False))

        # episode finished; determine crash/failure from last step's info
        crash = int(bool(info.get('fallen', False) or info.get('collided', False) or info.get('base_collision', False) or info.get('thigh_collision', False) or info.get('stuck', False)))
        crashes.append(crash)
        crash_type = 'fallen' if info.get('fallen', False) else 'collided' if info.get('collided', False) else 'base_collision' if info.get('base_collision', False) else 'thigh_collision' if info.get('thigh_collision', False) else 'stuck' if info.get('stuck', False) else 'none'
        print(f"episode {i+1}/{n} steps={steps} crash={crash}")

        # label and append per-episode data to global training buffers
        if training_out is not None:
            ep_obs = env.trainer.training_data['obs']
            ep_actions = env.trainer.training_data['actions']
            ep_dones = env.trainer.training_data['dones']
            ep_dones[-1] = True
            ep_rewards = env.trainer.training_data['rewards']
            ep_useful = env.trainer.training_data['useful']
            ep_weights = env.trainer.training_data['weights']
            training_data_all['obs'].extend(ep_obs)
            training_data_all['actions'].extend(ep_actions)
            training_data_all['dones'].extend(ep_dones)
            training_data_all['rewards'].extend(ep_rewards)
            training_data_all['useful'].extend(ep_useful)
            training_data_all['weights'].extend(ep_weights)

        if (i + 1) % args.log_interval == 0:
            Mean, RHF, Val = calculate_val(crashes)
            print(f"episode {i+1}/{n}  samples={len(crashes)}  Mean={Mean[-1]:.6f}  RHF={RHF[-1]:.6f}")

        if (i + 1) % args.save_interval == 0:
            if out_path:
                np.save(os.path.join(out_path, f'nde_{args.worker_id}.npy'), np.array(crashes, dtype=np.float32))
            # also save criticality buffers periodically
            if training_out is not None:
                try:
                    np.save(os.path.join(training_out, f'training_{args.worker_id}.npy'), np.array(training_data_all))
                    print(f'Wrote training data to {training_out} (samples={len(training_data_all)})')
                except Exception as e:
                    print('Failed to save training data:', e)

    # final stats
    Mean, RHF, Val = calculate_val(crashes)
    print("DONE")
    print(f"samples={len(crashes)}  Mean={Mean[-1]:.6f}  RHF={RHF[-1]:.6f}")

    # final save
    if out_path:
        np.save(os.path.join(out_path, f'nde_{args.worker_id}.npy'), np.array(crashes, dtype=np.float32))

    # final save training data
    if training_out is not None:
        try:
            np.save(os.path.join(training_out, f'training_{args.worker_id}.npy'), np.array(training_data_all))
            print(f'Wrote training data to {training_out} (samples={len(training_data_all)})')
        except Exception as e:
            print('Failed to save training data:', e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--controller_path', type=str, default='training/models/actor_init.zip', help='Path to SB3 .zip or .pt file containing the trained policy')
    parser.add_argument('--controller_path', type=str, default='training/models/run_offline1_new/run_offline1_new_offline_ep50.policy.pt')
    parser.add_argument('--critical_threshold', type=float, default=0.8, help='Criticality threshold (default: 0.5)')
    parser.add_argument('--worker_id', type=int, default=0)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--max_steps', type=int, default=40)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--out', type=str, default='training/results', help='Path to save crashes numpy array')
    parser.add_argument('--save_interval', type=int, default=10, help='Save results every N episodes at log interval')
    parser.add_argument('--training_out', type=str, default=None, help='Optional path to save criticality dataset (obs, actions, labels)')
    parser.add_argument('--nade', action='store_true')
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.training_out:
        os.makedirs(args.training_out, exist_ok=True)
    run(args)
