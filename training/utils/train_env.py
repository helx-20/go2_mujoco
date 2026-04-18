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
import mujoco
import gym
from typing import Tuple
from deploy_mujoco.utils import pd_control
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
        self.terrain_action_shape = (4,)
        low = np.full((4,), -1.0, dtype=np.float32)
        high = np.full((4,), 1.0, dtype=np.float32)

        self.terrain_action_edges = [np.linspace(low[d], high[d], num=11) for d in range(low.shape[0])]

        self.max_episode_steps = max_episode_steps
        self._step_count = 0
        # controller / terrain timing
        self.control_decimation = int(getattr(trainer, 'control_decimation', 1))
        self.terrain_decimation = int(getattr(trainer, 'terrain_decimation', 1) or 1)
        self.total_sim_steps = int(self.terrain_decimation * self.control_decimation)
        # adjust reward scales to account for internal terrain sampling frequency (divide non-success/collision rewards by terrain_decimation)
        for k in list(self.trainer.reward_scales.keys()):
            if k not in ['success', 'failed']:
                self.trainer.reward_scales[k] = float(self.trainer.reward_scales[k]) / self.terrain_decimation
        # internal counters and current terrain action
        self._sim_since_terrain = self.total_sim_steps
        self._current_terrain_action = np.zeros(self.terrain_action_shape, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        res = self.trainer.reset()
        obs = res[:-4] # exclude last terrain action from observation
        info = {}

        self._step_count = 0
        # reset timing counters so terrain will be sampled on first step
        self._sim_since_terrain = self.total_sim_steps
        self._current_terrain_action = np.zeros(self.terrain_action_shape, dtype=np.float32)
        return obs, info

    def step(self, controller_action) -> Tuple[np.ndarray, float, bool, bool, dict]:
        # Ensure arrays
        controller_action = np.asarray(controller_action, dtype=np.float32)

        # If it's time to update terrain, sample and apply a new terrain action
        sampled_terrain = False
        if self._sim_since_terrain >= self.total_sim_steps:
            bins = np.random.randint(0, 10, size=self.terrain_action_shape)

            centers = np.zeros(self.terrain_action_shape, dtype=np.float32)
            flat_bins = np.asarray(bins).reshape(-1)
            for d in range(flat_bins.shape[0]):
                b = int(flat_bins[d])
                e = self.terrain_action_edges[d]
                centers.flat[d] = 0.5 * (e[b] + e[b + 1])

            self._current_terrain_action = centers
            # apply terrain action
            try:
                self.trainer.terrain_changer.apply_action_vector(self._current_terrain_action)
                self.trainer.terrain_changer._refresh_terrain_safe()
            except Exception:
                pass
            if getattr(self.trainer, 'render', False):
                try:
                    self.trainer.viewer.update_hfield(self.trainer.terrain_changer.hfield_id)
                    self.trainer.viewer.sync()
                except Exception:
                    pass

            self._sim_since_terrain = 0
            sampled_terrain = True

        # Set controller previous action for observations
        self.trainer.go2_controller.action_policy_prev = controller_action.copy()

        # compute target dof positions from provided controller action (policy-scale -> dof pos)
        action_scale = float(getattr(self.trainer.go2_controller, 'action_scale', 1.0))
        target_dof_pos = controller_action * action_scale + self.trainer.go2_controller.default_angles

        # run low-level simulation for one controller interval
        for sim_i in range(int(self.control_decimation)):
            self.trainer.robot_counter += 1
            step_start = time.time()

            num_j = int(getattr(self.trainer.go2_controller, 'num_actions', 12))

            qpos_j = self.trainer.data.qpos[7:7 + num_j]
            qvel_j = self.trainer.data.qvel[6:6 + num_j]
            tau = pd_control(target_dof_pos, qpos_j, getattr(self.trainer.go2_controller, 'kps'), np.zeros_like(getattr(self.trainer.go2_controller, 'kds')), qvel_j, getattr(self.trainer.go2_controller, 'kds'))

            # write controls and step
            self.trainer.data.ctrl[:] = tau
            mujoco.mj_step(self.trainer.model, self.trainer.data)

            # rendering / realtime sync if available on trainer
            try:
                if getattr(self.trainer, 'render', False):
                    if getattr(self.trainer, 'lock_camera', False):
                        self.trainer.viewer.cam.lookat[:] = self.trainer.data.qpos[:3]
                    self.trainer.viewer.sync()

                if getattr(self.trainer, 'realtime_sim', False):
                    time_until_next_step = self.trainer.model.opt.timestep - (time.time() - step_start)
                    if time_until_next_step > 0:
                        time.sleep(time_until_next_step)
            except Exception:
                pass

        # advance sim counter
        self._sim_since_terrain += int(self.control_decimation)

        # compute reward / done info at controller step frequency (instantaneous)
        total_reward, reward_info, done = self.trainer.compute_reward()

        # Update last action so observations can include it
        try:
            self.trainer.terrain_changer.last_action = np.asarray(self._current_terrain_action, dtype=np.float32)
        except Exception:
            pass

        next_obs = self.trainer.get_terrain_observation()
        next_obs = next_obs[:-4]

        self._step_count += 1
        truncated = False
        if self._step_count >= self.max_episode_steps * self.terrain_decimation:
            truncated = True

        terminated = bool(done)
        failed = bool(reward_info.get('fallen', False) or reward_info.get('collided', False) or reward_info.get('base_collision', False) or reward_info.get('thigh_collision', False) or reward_info.get('stuck', False))

        if terminated and (not truncated) and (not failed):
            info_success = True
            success_reward = self.trainer.reward_scales.get('success', 0.0)
        else:
            info_success = False
            success_reward = 0.0

        info = {
            'total_reward': float(total_reward),
            **(reward_info if isinstance(reward_info, dict) else {}),
        }
        if sampled_terrain:
            info['terrain_action'] = np.asarray(self._current_terrain_action, dtype=np.float32).tolist()

        info['success'] = info_success
        reward = float(total_reward) + float(success_reward)

        if done:
            print(f"Episode done at step {self._step_count} (terminated={terminated}, truncated={truncated}), total_reward={total_reward:.3f}, success_reward={success_reward:.3f}, info={info}")

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
