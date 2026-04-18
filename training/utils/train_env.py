#!/usr/bin/env python3
"""
TrainEnv: wrapper for training the Go2 controller with internal terrain sampling.

Step signature: step(controller_action)
 - controller_action: action taken by the Go2 controller (policy output)
 - terrain actions are sampled internally (uniform discrete bins -> continuous centers)

The rest of the reward/done logic delegates to the existing TestEnv implementation.
"""
import os, sys
import time
import numpy as np
import gym
from typing import Tuple
try:
    from gymnasium.spaces import Box
except Exception:
    from gym.spaces import Box

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from training.utils.test_env import TestEnv

class TrainEnv(gym.Env):
    """Gym wrapper around TestEnv for training a controller.

    Observation: terrain observation (robot state + last terrain action)
    Action: controller action vector (policy) passed into the Go2 controller
    Reward: terrain reward (collision/fall)
    """

    def __init__(self, trainer: TestEnv, max_episode_steps: int = 1000):
        super().__init__()
        self.trainer = trainer
        # Compatibility with Gymnasium / Stable-Baselines3 wrappers
        self.render_mode = None
        try:
            self.metadata = getattr(self, 'metadata', {})
        except Exception:
            self.metadata = {}

        obs_dim = trainer.get_terrain_observation().shape[0] - 4 # exclude last terrain action from observation space
        # controller action dim: try to read from go2_controller, fall back to 12
        ctrl_dim = int(getattr(getattr(trainer, 'go2_controller', None), 'num_actions', 12))

        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = Box(low=-1.0, high=1.0, shape=(ctrl_dim,), dtype=np.float32)

        # Build terrain action discretization edges using trainer-provided
        # dimensions when available. Default to [-1, 1] per-dimension which
        # matches TerrainGymEnv's typical action space.
        act_dim = int(getattr(trainer, 'total_action_dims', 0) or 0)
        self.terrain_action_shape = (act_dim,) if act_dim > 0 else (0,)
        if act_dim > 0:
            low = np.full((act_dim,), -1.0, dtype=np.float32)
            high = np.full((act_dim,), 1.0, dtype=np.float32)
        else:
            low = np.zeros((0,), dtype=np.float32)
            high = np.zeros((0,), dtype=np.float32)

        self.terrain_action_edges = [np.linspace(low[d], high[d], num=11) for d in range(low.shape[0])]

        self.max_episode_steps = max_episode_steps
        self._step_count = 0

    def reset(self, *, seed=None, options=None):
        res = self.trainer.reset()
        obs = res[:-4] # exclude last terrain action from observation
        info = {}

        self._step_count = 0
        return obs, info

    def step(self, controller_action) -> Tuple[np.ndarray, float, bool, bool, dict]:
        # sample terrain action uniformly in discrete bins (10 bins per dim)
        if isinstance(self.terrain_action_shape, tuple) and len(self.terrain_action_shape) == 0:
            bins = np.random.randint(0, 10, size=(0,))
        else:
            bins = np.random.randint(0, 10, size=self.terrain_action_shape)

        centers = np.zeros(self.terrain_action_shape, dtype=np.float32)
        flat_bins = np.asarray(bins).reshape(-1)
        for d in range(flat_bins.shape[0]):
            b = int(flat_bins[d])
            e = self.terrain_action_edges[d]
            centers.flat[d] = 0.5 * (e[b] + e[b + 1])

        terrain_action = centers

        # Set controller's previous policy action
        self.trainer.go2_controller.action_policy_prev = np.asarray(controller_action, dtype=np.float32)

        # Delegate stepping to trainer (applies terrain_action and runs controller loop)
        next_obs, act, reward, done, info = self.trainer.step(terrain_action)
        next_obs = next_obs[:-4] # exclude last terrain action from observation

        self._step_count += 1
        truncated = False
        if self._step_count >= self.max_episode_steps:
            truncated = True

        terminated = bool(done)

        if terminated and (not truncated):
            info['success'] = True
            success_reward = self.trainer.reward_scales.get('success', 0.0)
        else:
            info['success'] = False
            success_reward = 0.0
        reward = float(reward) + float(success_reward)

        return next_obs, float(reward), bool(terminated), bool(truncated), info

    def render(self, mode='human'):
        # trainer handles rendering via its `render` flag and external viewer
        pass

    # Compatibility helper used by VecEnv wrappers (Gymnasium / SB3)
    def get_wrapper_attr(self, name: str):
        if hasattr(self, name):
            return getattr(self, name)
        raise AttributeError(name)


if __name__ == "__main__":
    print("TrainEnv: import-only module")
