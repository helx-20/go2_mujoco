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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import yaml
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from training.controller_env import ControllerEnv

# Silence Gym -> Gymnasium migration warning and noisy gym/gymnasium loggers
import warnings
import logging
warnings.filterwarnings("ignore", message=".*Please upgrade to Gymnasium.*", category=UserWarning)
logging.getLogger("gym").setLevel(logging.ERROR)
logging.getLogger("gymnasium").setLevel(logging.ERROR)


def make_env_fn(go2_cfg, max_episode_steps, idx, render_mode=False):
    def _thunk():
        env = ControllerEnv(go2_cfg, max_episode_steps=max_episode_steps, render_mode=render_mode)
        env = Monitor(env)
        return env
    return _thunk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='deploy_mujoco/terrain/configs/go2.yaml',
                        help='Path to go2 config YAML (default points to deploy_mujoco/terrain/configs/go2.yaml)')
    parser.add_argument('--total_timesteps', type=int, default=200000)
    parser.add_argument('--num_envs', type=int, default=16)
    parser.add_argument('--sim_device', type=str, default='cpu')
    parser.add_argument('--rl_device', type=str, default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_steps', type=int, default=40)
    parser.add_argument('--out', type=str, default='training/model')
    parser.add_argument('--run_name', type=str, default='run')
    parser.add_argument('--pretrain', type=str, default='training/model/actor_init.zip',
                        help='Path to a pretrained PyTorch model or SB3 .zip to initialize policy (default uses training/model/actor_init.zip)')
    args = parser.parse_args()

    # Prefer explicit --config path
    cfg_filename = os.path.basename(args.config)
    go2_cfg = ("terrain", cfg_filename)
    os.makedirs(args.out, exist_ok=True)

    # create vec env
    # create envs with rendering disabled by default
    env_fns = [make_env_fn(go2_cfg, args.max_steps, i, render_mode=False) for i in range(args.num_envs)]
    if args.num_envs == 1:
        vec_env = DummyVecEnv(env_fns)
    else:
        vec_env = SubprocVecEnv(env_fns)

    # If a pretrained SB3 .zip is provided, load it directly with the created env
    model = None
    if args.pretrain and os.path.isfile(args.pretrain) and args.pretrain.endswith('.zip'):
        try:
            from stable_baselines3 import PPO as SB3PPO
            print('Loading SB3 pretrained model directly from', args.pretrain)
            model = SB3PPO.load(args.pretrain, env=vec_env, device=args.rl_device)
            print('Loaded SB3 model and attached env — continuing training from pretrain .zip')
        except Exception as e:
            print('Failed to load SB3 .zip as model; falling back to fresh model. Error:', e)

    # Otherwise create a fresh model and try to copy compatible weights if available
    if model is None:
        hidden_sizes = [512, 256, 128]
        policy_kwargs = {'net_arch': dict(pi=hidden_sizes, vf=hidden_sizes), 'activation_fn': torch.nn.ELU}
        model = PPO('MlpPolicy', vec_env, verbose=1, device=args.rl_device, seed=args.seed, policy_kwargs=policy_kwargs)

        if args.pretrain and os.path.isfile(args.pretrain):
            try:
                from stable_baselines3 import PPO as SB3PPO
                sb3_model = SB3PPO.load(args.pretrain, device='cpu')

                # Source and destination state dicts
                src_sd = sb3_model.policy.state_dict()
                policy_sd = model.policy.state_dict()

                matched = {}
                used_src = set()

                # 1) Exact name + shape matches
                for src_k, src_v in src_sd.items():
                    if src_k in policy_sd and tuple(policy_sd[src_k].shape) == tuple(src_v.shape):
                        try:
                            policy_sd[src_k].copy_(src_v)
                            matched[src_k] = src_k
                            used_src.add(src_k)
                        except Exception:
                            pass

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
                            try:
                                policy_sd[dst_k].copy_(src_v)
                                matched[dst_k] = src_k
                                used_src.add(src_k)
                                break
                            except Exception:
                                continue

                # 3) Shape-only matching for any remaining dst keys (first-fit)
                for dst_k in list(policy_sd.keys()):
                    if dst_k in matched:
                        continue
                    dst_shape = tuple(policy_sd[dst_k].shape)
                    for src_k, src_v in src_sd.items():
                        if src_k in used_src:
                            continue
                        if tuple(src_v.shape) == dst_shape:
                            try:
                                policy_sd[dst_k].copy_(src_v)
                                matched[dst_k] = src_k
                                used_src.add(src_k)
                                break
                            except Exception:
                                continue

                model.policy.load_state_dict(policy_sd)
                print(f'Initialized policy from {args.pretrain} — matched {len(matched)} tensors')
            except Exception as e:
                print('Pretrain load attempt failed:', e)

    model.learn(total_timesteps=int(args.total_timesteps))

    save_path = os.path.join(args.out, f'{args.run_name}_ppo.zip')
    model.save(save_path)
    print('Saved model to', save_path)

    # # attempt to export the learned policy to TorchScript for use in Go2Controller
    # try:
    #     export_dir = os.path.join(args.out, 'jit')
    #     os.makedirs(export_dir, exist_ok=True)
    #     # reload into cpu
    #     model_cpu = PPO.load(save_path, device='cpu')
    #     policy = model_cpu.policy

    #     import torch as th

    #     class TraceWrapper(th.nn.Module):
    #         def __init__(self, pol):
    #             super().__init__()
    #             # store the policy (ActorCriticPolicy)
    #             self.pol = pol

    #         def forward(self, x: th.Tensor) -> th.Tensor:
    #             # forward through the SB3 policy to get actions
    #             # policy.forward returns (actions, values, log_prob)
    #             actions, _, _ = self.pol.forward(x)
    #             return actions

    #     # set policy to eval mode to reduce nondeterministic tracing behavior
    #     try:
    #         policy.to('cpu')
    #         policy.eval()
    #     except Exception:
    #         pass
    #     wrapper = TraceWrapper(policy).cpu()

    #     # create an example input from observation space
    #     try:
    #         obs_space = vec_env.observation_space
    #         sample = obs_space.sample()
    #         if isinstance(sample, np.ndarray):
    #             example = th.from_numpy(sample.astype(np.float32)).unsqueeze(0)
    #         else:
    #             # fallback to zeros
    #             example = th.zeros((1, int(obs_space.shape[0])), dtype=th.float32)
    #     except Exception:
    #         example = th.zeros((1, policy.observation_space.shape[0]), dtype=th.float32) if hasattr(policy, 'observation_space') else th.zeros((1, 48), dtype=th.float32)

    #     # trace the wrapper (more robust than scripting for complex SB3 modules)
    #     # Trace without strict trace checking to avoid failures from stochastic nodes
    #     traced = th.jit.trace(wrapper, example, check_trace=False)
    #     jit_path = os.path.join(export_dir, f'{args.run_name}_policy.jit')
    #     traced.save(jit_path)
    #     print('Exported TorchScript (traced) policy to', jit_path)
    # except Exception as e:
    #     print('Failed to export TorchScript policy:', e)


if __name__ == '__main__':
    main()
