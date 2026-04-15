#!/usr/bin/env python3
"""Simplified transformer: load a Go2 TorchScript policy (or state_dict) and
map its numeric MLP layers into a Stable-Baselines3 `PPO` MlpPolicy with a
3-layer ELU network, then save as an SB3 `.zip` for fine-tuning.

Usage:
    python training/utils/transform_actor_model_to_sb3.py --actor-pth training/model/go2_controller_new.pt --out training/model/actor_init.zip
"""

from __future__ import annotations
import os
import sys
import argparse

# ensure repo root on path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Prefer gymnasium; fallback to gym
try:
    import gymnasium as gym
    from gymnasium.spaces import Box
except Exception:
    import gym
    from gym.spaces import Box


class DummyEnv(gym.Env):
    """Gym/Gymnasium-compatible dummy env for SB3 policy initialization."""

    def __init__(self, obs_dim=48, act_dim=12):
        super().__init__()
        self.observation_space = Box(low=-float('inf'), high=float('inf'), shape=(obs_dim,), dtype=np.float32)
        self.action_space = Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        return self.observation_space.sample(), {}

    def step(self, a):
        return self.observation_space.sample(), 0.0, False, False, {}


def load_actor_state(actor_pth: str):
    """Load a TorchScript module or a state_dict and return a flat dict of tensors."""
    if actor_pth.endswith('.pt') or actor_pth.endswith('.jit'):
        try:
            m = torch.jit.load(actor_pth, map_location='cpu')
            if hasattr(m, 'state_dict'):
                sd = m.state_dict()
            else:
                sd = {k: v for k, v in m.named_parameters()} if hasattr(m, 'named_parameters') else {}
        except Exception:
            sd = torch.load(actor_pth, map_location='cpu')
    else:
        sd = torch.load(actor_pth, map_location='cpu')

    if isinstance(sd, dict):
        return {k: v for k, v in sd.items()}
    return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--actor-pth', type=str, default='pre_train/go2/go2_controller.pt')
    parser.add_argument('--out', type=str, default='training/model/actor_init.zip')
    parser.add_argument('--obs-dim', type=int, default=48)
    parser.add_argument('--act-dim', type=int, default=12)
    args = parser.parse_args()

    if not os.path.exists(args.actor_pth):
        print('Actor file not found:', args.actor_pth)
        return

    actor_sd = load_actor_state(args.actor_pth)
    if not actor_sd:
        print('Failed to read actor state from', args.actor_pth)
        return

    import re
    layers = {}
    for k, v in actor_sd.items():
        m = re.match(r'^(\d+)\.(weight|bias)$', k)
        if m:
            lid = int(m.group(1))
            layers.setdefault(lid, {})[m.group(2)] = v

    if not layers:
        print('No numeric MLP layers found in actor state dict. Keys sample:')
        for kk in list(actor_sd.keys())[:20]:
            print(' ', kk)
        return

    layer_ids = sorted(layers.keys())
    print('Found numeric actor layers:', layer_ids)

    hidden_sizes = []
    for lid in layer_ids[:-1]:
        w = layers[lid].get('weight')
        if w is not None and w.ndim == 2:
            hidden_sizes.append(int(w.shape[0]))

    if len(hidden_sizes) >= 3:
        hidden_sizes = hidden_sizes[:3]
    else:
        while len(hidden_sizes) < 3:
            hidden_sizes.append(hidden_sizes[-1] if hidden_sizes else 128)

    print('Using hidden sizes for SB3 policy:', hidden_sizes)

    policy_kwargs = {'net_arch': dict(pi=hidden_sizes, vf=hidden_sizes), 'activation_fn': torch.nn.ELU}

    vec = DummyVecEnv([lambda: DummyEnv(obs_dim=args.obs_dim, act_dim=args.act_dim)])
    model = PPO('MlpPolicy', vec, verbose=0, device='cpu', policy_kwargs=policy_kwargs)

    policy_sd = model.policy.state_dict()

    p_layers = []
    for k in policy_sd.keys():
        m = re.match(r'^mlp_extractor\.policy_net\.(\d+)\.(weight|bias)$', k)
        if m:
            p_layers.append(int(m.group(1)))
    p_layers = sorted(set(p_layers))

    for i, lid in enumerate(layer_ids[:-1]):
        if i >= len(p_layers):
            break
        w = layers[lid].get('weight')
        b = layers[lid].get('bias')
        if w is not None:
            pk = f'mlp_extractor.policy_net.{p_layers[i]}.weight'
            if tuple(policy_sd[pk].shape) == tuple(w.shape):
                policy_sd[pk].copy_(w)
                print('Mapped actor layer', lid, '->', pk)
        if b is not None:
            pkb = f'mlp_extractor.policy_net.{p_layers[i]}.bias'
            if tuple(policy_sd[pkb].shape) == tuple(b.shape):
                policy_sd[pkb].copy_(b)

    last_id = layer_ids[-1]
    last_w = layers[last_id].get('weight')
    last_b = layers[last_id].get('bias')
    if last_w is not None and 'action_net.weight' in policy_sd:
        if tuple(policy_sd['action_net.weight'].shape) == tuple(last_w.shape):
            policy_sd['action_net.weight'].copy_(last_w)
            print('Mapped actor last weight -> action_net.weight')
        else:
            if last_w.ndim == 2 and tuple(policy_sd['action_net.weight'].shape) == tuple(last_w.T.shape):
                policy_sd['action_net.weight'].copy_(last_w.T)
                print('Mapped actor last weight (transposed) -> action_net.weight')
    if last_b is not None and 'action_net.bias' in policy_sd and tuple(policy_sd['action_net.bias'].shape) == tuple(last_b.shape):
        policy_sd['action_net.bias'].copy_(last_b)

    model.policy.load_state_dict(policy_sd)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    model.save(args.out)
    print('Saved initialized SB3 model to', args.out)


if __name__ == '__main__':
    main()