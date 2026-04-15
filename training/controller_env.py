"""Gym environment wrapper to train Go2 controller in MuJoCo.

This environment presents the controller observation/action interface
compatible with `deploy_mujoco/terrain/go2_controller.py` but accepts
policy actions directly (instead of loading a jit policy). It runs the
low-level PD controller and provides a simple reward signal (speed
tracking minus fall/collision penalties). This is a minimal viable env
to train the Go2 controller with standard RL libraries (PPO/SAC).
"""
from typing import Tuple
import os
import time
import sys

import numpy as np
try:
    import mujoco
except Exception as e:
    raise ImportError(
        "MuJoCo Python package not found.\n"
        "Install it in your active environment (e.g. `pip install mujoco`) and ensure the MuJoCo"
        " binaries and license are available. Original error: {}".format(e)
    ) from e
try:
    # suppress the Gym->Gymnasium migration warning emitted by older gym versions
    import warnings
    import logging
    warnings.filterwarnings("ignore", message=".*Please upgrade to Gymnasium.*", category=UserWarning)
    logging.getLogger("gym").setLevel(logging.ERROR)
    logging.getLogger("gymnasium").setLevel(logging.ERROR)

    import gymnasium as gym
    from gymnasium.spaces import Box
except Exception:
    import gym
    from gym.spaces import Box

from deploy_mujoco.utils import pd_control, quat_to_rpy, get_gravity_orientation


class ControllerEnv(gym.Env):
    """Env where actions are controller policy outputs (shape num_actions).

    Observation shape follows `Go2Controller.get_observation` layout.
    """
    def __init__(self, go2_cfg: Tuple[str, str], max_episode_steps: int = 1000, render_mode: bool = False):
        # go2_cfg: (dir, config_yaml)
        cfg_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../deploy_mujoco/terrain")
        # import local config loader for compatibility
        import yaml
        cfg_path = os.path.join(cfg_dir, "configs", go2_cfg[1])
        with open(cfg_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)

        self.cfg = cfg
        # store render_mode for compatibility; env does not perform visualization when False
        self.render_mode = bool(render_mode)
        self.xml_path = cfg["xml_path"]
        # MuJoCo model and data
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = float(cfg.get("simulation_dt", 0.002))

        self.control_decimation = int(cfg.get("control_decimation", 10))
        self.kps = np.array(cfg.get("kps", []), dtype=np.float32)
        self.kds = np.array(cfg.get("kds", []), dtype=np.float32)
        self.default_angles = np.array(cfg.get("default_angles", []), dtype=np.float32)
        self.num_actions = int(cfg.get("num_actions", 12))
        self.action_scale = float(cfg.get("action_scale", 0.25))
        self.lin_vel_scale = float(cfg.get("lin_vel_scale", 1.0))
        self.dof_pos_scale = float(cfg.get("dof_pos_scale", 1.0))
        self.dof_vel_scale = float(cfg.get("dof_vel_scale", 1.0))

        # observation dim (match Go2Controller.get_observation)
        self.obs_dim = int(cfg.get("num_obs", 48))
        self.action_space = Box(low=-1.0, high=1.0, shape=(self.num_actions,), dtype=np.float32)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)

        self.max_episode_steps = int(max_episode_steps)
        self._step_count = 0

        # load reward scales from legged_gym go2_stair config if available
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'training_code_base', 'legged_gym'))
            from training.utils.go2_terrain_config import GO2TerrainCfg
            rcfg = GO2TerrainCfg()
            # convert to dict-like reward scales if present
            self.reward_scales = getattr(rcfg, 'rewards').scales.__dict__ if hasattr(rcfg.rewards, 'scales') else {}
        except Exception:
            # fallback: minimal scales
            self.reward_scales = {}

        # prepare foot/body ids for contact-based rewards
        self.foot_body_ids = []
        for i in range(self.model.nbody):
            try:
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i).decode() if isinstance(mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i), bytes) else mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            except Exception:
                name = ''
            if 'foot' in name.lower():
                self.foot_body_ids.append(int(i))

        # buffers used by several rewards
        self.last_contacts = np.zeros((len(self.foot_body_ids),), dtype=bool)
        self.feet_air_time = np.zeros((len(self.foot_body_ids),), dtype=np.float32)
        self.last_actions = np.zeros((self.num_actions,), dtype=np.float32)
        self.last_dof_vel = np.zeros((len(self.default_angles),), dtype=np.float32) if self.default_angles.size>0 else np.zeros((self.num_actions,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        mujoco.mj_resetData(self.model, self.data)
        self._step_count = 0
        # warmup initial steps with PD to settle
        target_dof_pos = self.default_angles.copy()
        total_sim_steps = int(self.control_decimation * 10)
        for sim_i in range(total_sim_steps):
            tau = pd_control(target_dof_pos, self.data.qpos[7:], self.kps, np.zeros_like(self.kds), self.data.qvel[6:], self.kds)
            self.data.ctrl[:] = tau
            mujoco.mj_step(self.model, self.data)
        return self._get_obs(), {}

    def get_wrapper_attr(self, name):
        """Compatibility helper for Stable-Baselines3/SubprocVecEnv remote calls.

        SB3 may call `env.get_wrapper_attr(name)` in worker processes to
        retrieve attributes from wrapped envs. Support both single string
        and list/tuple of attribute names.
        """
        if isinstance(name, (list, tuple)):
            return [getattr(self, n) for n in name]
        return getattr(self, name)

    def step(self, action):
        # action is policy output in [-1,1]^num_actions
        action = np.asarray(action, dtype=np.float32)
        # compute target dof pos
        target_dof_pos = action * self.action_scale + self.default_angles

        # run PD controller for control_decimation sim steps
        for sim_i in range(self.control_decimation):
            tau = pd_control(target_dof_pos, self.data.qpos[7:], self.kps, np.zeros_like(self.kds), self.data.qvel[6:], self.kds)
            self.data.ctrl[:] = tau
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1

        obs = self._get_obs()
        reward, info, done = self._compute_reward_and_info(action)
        truncated = False
        if self._step_count >= self.max_episode_steps:
            truncated = True
            done = True

        return obs, float(reward), bool(done), bool(truncated), info

    def _get_obs(self):
        # compose observation similar to Go2Controller.get_observation
        qj = self.data.qpos[7:].copy()
        dqj = self.data.qvel[6:].copy()
        quat = self.data.qpos[3:7].copy()
        lin_vel = self.data.qvel[:3].copy()
        ang_vel = self.data.qvel[3:6].copy()

        qj = (qj - self.default_angles) * self.dof_pos_scale
        dqj = dqj * self.dof_vel_scale

        gravity_orientation = get_gravity_orientation(quat)
        lin_vel = lin_vel * self.lin_vel_scale
        ang_vel = ang_vel * self.cfg.get("ang_vel_scale", 1.0)

        obs = np.zeros(self.obs_dim, dtype=np.float32)
        obs[:3] = lin_vel
        obs[3:6] = ang_vel
        obs[6:9] = gravity_orientation
        # cmd placeholder
        obs[9:12] = np.array(self.cfg.get("cmd_init", [0,0,0]), dtype=np.float32)
        obs[12: 12 + self.num_actions] = qj
        obs[12 + self.num_actions: 12 + 2 * self.num_actions] = dqj
        obs[12 + 2 * self.num_actions: 12 + 3 * self.num_actions] = self.last_actions
        return obs

    def _compute_reward_and_info(self, action):
        # compute a set of reward terms inspired by the IsaacGym go2_stair task
        base_z = float(self.data.qpos[2])
        lin_vel = float(np.linalg.norm(self.data.qvel[:2]))
        quat = self.data.qpos[3:7]
        roll, pitch, _ = quat_to_rpy(quat)

        # contacts processing
        collided = False
        contact_forces = []
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, i, force)
            f_mag = np.linalg.norm(force[:3])
            contact_forces.append(f_mag)
            if f_mag > 5.0:
                collided = True

        # fallen check via base height and angles
        fallen = base_z < 0.15 or abs(roll) > 1.0 or abs(pitch) > 0.8

        # Reward terms
        reward = 0.0
        # tracking linear velocity (approx using forward x vel)
        target_lin = float(self.cfg.get('cmd_init', [1.0, 0, 0])[0])
        tracking_lin = np.exp(- (target_lin - float(self.data.qvel[0]))**2 / max(1e-6, 0.1))
        reward += float(self.reward_scales.get('tracking_lin_vel', 1.0) * tracking_lin)
        # tracking angular vel (yaw)
        target_ang = float(self.cfg.get('cmd_init', [0,0,0])[2])
        tracking_ang = np.exp(- (target_ang - float(self.data.qvel[5]))**2 / max(1e-6, 0.1))
        reward += float(self.reward_scales.get('tracking_ang_vel', 1.0) * tracking_ang)

        # torque penalty
        # approximate torques via pd_control call
        target_dof_pos = action * self.action_scale + self.default_angles
        tau = pd_control(target_dof_pos, self.data.qpos[7:], self.kps, np.zeros_like(self.kds), self.data.qvel[6:], self.kds)
        torque_pen = np.sum(np.square(tau))
        reward += float(self.reward_scales.get('torques', -0.0002) * torque_pen)

        # dof pos limits penalty
        dof_pos = np.asarray(self.data.qpos[7:])
        # assume default limits +-1.5 rad
        out_limits = np.sum(np.maximum(0.0, np.abs(dof_pos) - 2.0))
        reward += float(self.reward_scales.get('dof_pos_limits', -10.0) * out_limits)

        # collision / fallen penalties
        if fallen:
            reward += float(self.reward_scales.get('termination', -100.0))
        if collided:
            reward += float(self.reward_scales.get('collision', -1.0))

        # feet air time reward (simple): reward when foot contact reappears
        # build current contact flags per foot
        curr_contacts = []
        for fb in self.foot_body_ids:
            # check any contact involving this body
            has_contact = False
            for i in range(self.data.ncon):
                contact = self.data.contact[i]
                b1 = int(self.model.geom_bodyid[contact.geom1])
                b2 = int(self.model.geom_bodyid[contact.geom2])
                if b1 == fb or b2 == fb:
                    # check normal force
                    force = np.zeros(6)
                    mujoco.mj_contactForce(self.model, self.data, i, force)
                    if np.linalg.norm(force[:3]) > 1.0:
                        has_contact = True
                        break
            curr_contacts.append(has_contact)

        # update feet air time
        for i, contacted in enumerate(curr_contacts):
            if not contacted:
                self.feet_air_time[i] += self.model.opt.timestep * self.control_decimation
            else:
                # reward first contact after air time
                if self.feet_air_time[i] > 0.5:
                    reward += float(self.reward_scales.get('feet_air_time', 0.0) * (self.feet_air_time[i] - 0.5))
                self.feet_air_time[i] = 0.0

        # update last buffers
        self.last_actions = np.asarray(action, dtype=np.float32)
        self.last_dof_vel = np.asarray(self.data.qvel[6:], dtype=np.float32)

        info = {
            'fallen': fallen,
            'collided': collided,
            'speed': lin_vel,
            'base_height': base_z,
        }
        done = bool(fallen or collided)
        return float(reward), info, done
