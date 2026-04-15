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

    model = PPO('MlpPolicy', vec_env, verbose=1, device=args.rl_device, seed=args.seed)

    # If a pretrained controller is provided, attempt to load its weights into the SB3 policy
    if args.pretrain and os.path.isfile(args.pretrain):
        from stable_baselines3 import PPO as SB3PPO
        sb3_model = SB3PPO.load(args.pretrain, device='cpu')
        src_sd = sb3_model.policy.state_dict()

        # copy matching tensors into the current SB3 policy
        policy_sd = model.policy.state_dict()
        matched = 0
        for k, v in src_sd.items():
            if k in policy_sd and tuple(policy_sd[k].shape) == tuple(v.shape):
                try:
                    policy_sd[k].copy_(v)
                    matched += 1
                except Exception:
                    pass

        model.policy.load_state_dict(policy_sd)
        print(f'Initialized policy from {args.pretrain} — matched {matched} tensors')

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
