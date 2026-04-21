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
from training.utils.test_env import TestEnv, safe_call
import torch

class TrainEnv(gym.Env):
    """Gym wrapper around TestEnv for training a controller.

    Observation: terrain observation (robot state + last terrain action)
    Action: controller action vector (policy) passed into the Go2 controller
    Reward: terrain reward (collision/fall)
    """

    def __init__(self, trainer: TestEnv, max_episode_steps: int = 1000, nade=False, criticality_model=None):
        super().__init__()
        self.trainer = trainer
        # Compatibility with Gymnasium / Stable-Baselines3 wrappers
        self.render_mode = None
        self.nade = nade
        self.criticality_model = criticality_model

        try:
            self.metadata = getattr(self, 'metadata', {})
        except Exception:
            self.metadata = {}

        obs_dim = trainer.get_terrain_observation().shape[0] - 4 # exclude last terrain action from observation space
        # controller action dim: try to read from go2_controller, fall back to 12
        ctrl_dim = int(getattr(getattr(trainer, 'go2_controller', None), 'num_actions', 12))

        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = Box(low=-10.0, high=10.0, shape=(ctrl_dim,), dtype=np.float32)

        # Build terrain action discretization edges using trainer-provided
        # dimensions when available. Default to [-1, 1] per-dimension which
        # matches TerrainGymEnv's typical action space.
        self.terrain_action_shape = (4,)
        low = np.full((4,), -1.0, dtype=np.float32)
        high = np.full((4,), 1.0, dtype=np.float32)

        self.terrain_action_edges = [np.linspace(low[d], high[d], num=11) for d in range(low.shape[0])]
        D = 4
        grids = np.meshgrid(*[np.arange(10) for _ in range(D)], indexing='ij')
        bins_flat = np.stack([g.reshape(-1) for g in grids], axis=1).astype(np.int64)
        num_actions = bins_flat.shape[0]
        centers = np.zeros((num_actions, D), dtype=np.float32)
        for d in range(D):
            e = self.terrain_action_edges[d]
            b_idx = bins_flat[:, d]
            centers[:, d] = 0.5 * (e[b_idx] + e[b_idx + 1])
        self.candidates_arr = centers

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
        self._sim_since_terrain = 0
        self._current_terrain_action = np.zeros(self.terrain_action_shape, dtype=np.float32)
        self.trainer.terrain_changer.last_action = self._current_terrain_action.copy()
        self._controller_normal_action = np.zeros_like(self.action_space.shape, dtype=np.float32)
        self._current_criticality = 0.0

    def reset(self, *, seed=None, options=None):
        res = self.trainer.reset()
        self.trainer.go2_controller.cmd = self.trainer.go2_controller.update_command(self.trainer.data, self.trainer.go2_controller.cmd, self.trainer.go2_controller.heading_stiffness, self.trainer.go2_controller.heading_target, self.trainer.go2_controller.heading_command)
        obs = self.trainer.go2_controller.get_observation(self.trainer.data).astype(np.float32)
        info = {}

        self._step_count = 0
        # reset timing counters so terrain will be sampled on first step
        self._sim_since_terrain = 0 # self.total_sim_steps
        self._current_terrain_action = np.zeros(self.terrain_action_shape, dtype=np.float32)
        self.trainer.terrain_changer.last_action = self._current_terrain_action.copy()
        self._controller_normal_action = np.zeros_like(self.action_space.shape, dtype=np.float32)
        self._current_criticality = 0.0
        return obs, info

    def step(self, controller_action) -> Tuple[np.ndarray, float, bool, bool, dict]:
        # Ensure arrays
        controller_action = np.asarray(controller_action, dtype=np.float32)

        # If any candidate terrain action is predicted to be not critical, override the controller action with the normal (non-policy) action
        if self._current_criticality <= self.trainer.critical_threshold:
            controller_action = self._controller_normal_action.copy()

        # Keep the controller's recorded previous policy in sync with the
        # externally provided controller_action so observations match.
        self.trainer.go2_controller.action_policy_prev = controller_action.copy()

        # Compute target dof positions from provided controller action (policy-scale -> dof pos)
        try:
            action_scale = float(getattr(self.trainer.go2_controller, 'action_scale', 0.25))
        except Exception:
            action_scale = 0.25
        target_dof_pos = controller_action * action_scale + self.trainer.go2_controller.default_angles

        # Run low-level simulation for one controller interval. At control
        # boundaries we use the externally provided target (zero-order hold).
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

        # If it's time to update terrain, sample and apply a new terrain action
        sampled_terrain = False
        if self._sim_since_terrain >= self.total_sim_steps:
            # compute criticality of all candidate terrain actions
            if self.criticality_model is not None:
                with torch.no_grad():
                    terrain_obs = self.trainer.get_terrain_observation().astype(np.float32)
                    t_obs = torch.from_numpy(terrain_obs.astype(np.float32)).unsqueeze(0).repeat(self.candidates_arr.shape[0], 1)
                    t_act = torch.from_numpy(np.asarray(self.candidates_arr, dtype=np.float32))
                    t_in = torch.cat([t_obs, t_act], dim=1)
                    t_out = self.criticality_model(t_in)
                    criticality = torch.nn.functional.softmax(t_out, dim=1)[:, 1].squeeze().cpu().numpy()
            else:
                criticality = np.zeros((self.candidates_arr.shape[0],), dtype=np.float32)

            # select terrain action
            if not self.nade:
                idx = np.random.randint(0, self.candidates_arr.shape[0])
            else:
                if np.max(criticality) > 3e-1:
                    idx = int(np.argmax(criticality))
                else:
                    idx = np.random.randint(0, self.candidates_arr.shape[0])

            self._current_criticality = float(criticality[idx])

            # store current terrain action
            centers = np.asarray(self.candidates_arr[idx], dtype=np.float32)
            self._current_terrain_action = centers
            # apply terrain action
            try:
                self.trainer.terrain_changer.apply_action_vector(self._current_terrain_action)
                self.trainer.terrain_changer._refresh_terrain_safe()
            except Exception:
                pass
            if getattr(self.trainer, 'render', False):
                self.trainer.viewer.update_hfield(self.trainer.terrain_changer.hfield_id)
                self.trainer.viewer.sync()

            self.trainer.terrain_changer.last_action = np.asarray(self._current_terrain_action, dtype=np.float32)

            self._sim_since_terrain = 0
            sampled_terrain = True

        # compute reward / done info at controller step frequency (instantaneous)
        total_reward, reward_info, done = self.trainer.compute_reward()
        
        self.trainer.go2_controller.cmd = self.trainer.go2_controller.update_command(self.trainer.data, self.trainer.go2_controller.cmd, self.trainer.go2_controller.heading_stiffness, self.trainer.go2_controller.heading_target, self.trainer.go2_controller.heading_command)
        next_obs = self.trainer.go2_controller.get_observation(self.trainer.data).astype(np.float32)

        self._step_count += 1
        truncated = False
        if self._step_count >= self.max_episode_steps * self.terrain_decimation:
            truncated = True

        terminated = bool(done)
        done = bool(done or truncated)
        failed = bool(reward_info.get('fallen', False) or reward_info.get('collided', False) or reward_info.get('base_collision', False) or reward_info.get('thigh_collision', False) or reward_info.get('stuck', False))

        if done and (not truncated) and (not failed):
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

        self._controller_normal_action = self.trainer.go2_controller.policy(torch.tensor(next_obs)).detach().cpu().numpy()

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
