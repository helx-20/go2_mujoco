"""Gym environment wrapper to train Go2 controller in MuJoCo.

This environment presents the controller observation/action interface
compatible with `training/utils/go2_controller_test.py` but accepts
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
from deploy_mujoco.terrain_params import TerrainChanger
import torch


class ControllerEnv(gym.Env):
    """Env where actions are controller policy outputs (shape num_actions).

    Observation shape follows `Go2Controller.get_observation` layout.
    """
    def __init__(self, warmup_policy, config_file_path: str, max_episode_steps: int = 1000, render_mode: bool = False):
        # go2_cfg: (dir, config_yaml)
        cfg_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "utils")
        # import local config loader for compatibility
        import yaml
        cfg_path = os.path.join(cfg_dir, config_file_path)
        with open(cfg_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)

        self.cfg = cfg
        from training.utils.go2_controller_test import Go2Controller
        self._go2_controller = Go2Controller(config_file_path, warmup_policy)
        # Copy controller's canonical parameters to this env to avoid
        # accidental mismatches (action_scale, default_angles, kp/kd, scales, cmd, etc).
        try:
            ctrl = self._go2_controller
            for _attr in ['num_obs', 'policy_decimation', 'ang_vel_scale', 'heading_stiffness',
                          'heading_target', 'heading_command', 'cmd_scale', 'cmd', 'cmd_init',
                          'kps', 'kds', 'default_angles', 'action_scale', 'lin_vel_scale',
                          'dof_pos_scale', 'dof_vel_scale', 'num_actions']:
                if hasattr(ctrl, _attr):
                    setattr(self, _attr, getattr(ctrl, _attr))
        except Exception:
            pass
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

        # action/observation spaces - observation_space will be computed later
        self.action_space = Box(low=-1.0, high=1.0, shape=(self.num_actions,), dtype=np.float32)
        self.observation_space = None

        self.max_episode_steps = int(max_episode_steps)
        self._step_count = 0

        # initialization/warmup time to match TestEnv's protective startup (seconds + frames)
        self.init_skip_time = float(self.cfg.get('init_skip_time', 2.0))
        self.init_skip_frame = int(self.cfg.get('init_skip_frame', 10))

        # load reward scales and some config from legged_gym go2_stair config if available
        self.rcfg = None
        self.reward_scales = {}
        self.only_positive_rewards = False
        self.tracking_sigma = 0.1
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), 'training', 'utils', 'legged_gym'))
            try:
                from training.utils.go2_terrain_config import GO2TerrainCfg
            except Exception:
                # either legged_gym not available or provided legged_gym is missing expected
                # attributes (partial copy). Create a minimal stub so the config module can
                # import its base classes and be evaluated.
                import types
                lr_mod = types.ModuleType('legged_gym.envs.base.legged_robot_config')
                class LeggedRobotCfg:
                    class init_state: pass
                    class control: pass
                    class asset: pass
                    class rewards:
                        class scales: pass
                    class terrain: pass
                class LeggedRobotCfgPPO:
                    class algorithm: pass
                    class runner: pass
                lr_mod.LeggedRobotCfg = LeggedRobotCfg
                lr_mod.LeggedRobotCfgPPO = LeggedRobotCfgPPO
                sys.modules['legged_gym'] = types.ModuleType('legged_gym')
                sys.modules['legged_gym.envs'] = types.ModuleType('legged_gym.envs')
                sys.modules['legged_gym.envs.base'] = types.ModuleType('legged_gym.envs.base')
                sys.modules['legged_gym.envs.base.legged_robot_config'] = lr_mod
                # now import the user-provided GO2TerrainCfg
                from training.utils.go2_terrain_config import GO2TerrainCfg

            rcfg = GO2TerrainCfg()
            self.rcfg = rcfg
            # convert to dict-like reward scales if present
            try:
                scales_obj = getattr(rcfg, 'rewards').scales
                # scales may be a class; take non-dunder attributes
                self.reward_scales = {k: getattr(scales_obj, k) for k in dir(scales_obj) if not k.startswith('__')}
            except Exception:
                self.reward_scales = {}
            self.only_positive_rewards = getattr(rcfg.rewards, 'only_positive_rewards', False)
            self.tracking_sigma = getattr(rcfg.rewards, 'tracking_sigma', 0.1)
        except Exception as e:
            # fallback: minimal scales
            print('GO2TerrainCfg import/extract failed:', e)
            self.reward_scales = {}

        # load terrain config (for thresholds, termination rules)
        try:
            terrain_cfg_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'utils','terrain_config.yaml')
            import yaml as _yaml
            with open(terrain_cfg_path, 'r', encoding='utf-8') as _f:
                self.terrain_config = _yaml.safe_load(_f) or {}
        except Exception:
            self.terrain_config = {}

        # Terrain changer setup (so ControllerEnv can use same hfield generation/params)
        try:
            tcfg = self.terrain_config
            self.terrain_decimation = int(tcfg.get('terrain_action', {}).get('terrain_decimation', 1))
            terrain_types = tcfg.get('terrain_action', {}).get('terrain_types', [])
            # build action dims mapping similarly to TestEnv (ControllerEnv won't apply actions)
            action_dims = {}
            if 'bump' in terrain_types:
                action_dims['bump'] = 4
            if 'slide_friction' in terrain_types:
                action_dims['slide_friction'] = 1
            if 'solref' in terrain_types:
                action_dims['solref'] = 1
            # store terrain_action_dim based on action_dims so obs_space can be computed before reset
            self.terrain_action_dim = int(sum(action_dims.values())) if isinstance(action_dims, dict) and len(action_dims) > 0 else 0
            self.terrain_changer = TerrainChanger(self.model, self.data, action_dims=action_dims, config_file='terrain_config.yaml')
        except Exception:
            self.terrain_changer = None
            self.terrain_action_dim = 0

        # optional mapping from policy action order to model action order
        # some deploy configs expose `mapping_joints` which maps policy indices -> model indices
        try:
            mapping = cfg.get('mapping_joints', None)
            if mapping is not None:
                self.policy2model = np.array(mapping, dtype=np.int32)
            else:
                self.policy2model = None
        except Exception:
            self.policy2model = None

        # convenience getters for terrain thresholds/defaults
        ec = self.terrain_config.get('event_and_reward', {})
        term_cfg = self.terrain_config.get('termination', {})
        obs_cfg = self.terrain_config.get('observation', {})
        self._fall_height_threshold = float(ec.get('fall_height_threshold', 0.15))
        self._fall_angle_threshold = float(ec.get('fall_angle_threshold', 1.3))
        self._collision_force_threshold = float(ec.get('collision_force_threshold', 80.0))
        self._stuck_speed_threshold = float(ec.get('stuck_speed_threshold', 0.05))
        self._failure_flags = self.terrain_config.get('event_and_reward', {}).get('failure_flags', {})
        self._termination_cfg = term_cfg
        self._obs_contact_force_threshold = float(obs_cfg.get('contact_force_threshold', 1.0))

        # prepare foot/body ids for contact-based rewards
        self.foot_body_ids = []
        for i in range(self.model.nbody):
            try:
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i).decode() if isinstance(mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i), bytes) else mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            except Exception:
                name = ''
            if 'foot' in name.lower():
                self.foot_body_ids.append(int(i))

            # compute observation dimension now that foot/body ids and terrain params are initialized
            try:
                self.robot_obs_dim = 12 + 3 * self.num_actions
                foot_contact_dim = len(self.foot_body_ids) if self.obs_include_foot_contacts else 0
                local_map_dim = self.local_map_side * self.local_map_side if self.local_map_enabled else 0
                terrain_action_dim = int(getattr(self, 'terrain_action_dim', 0))
                self.obs_dim = int(self.robot_obs_dim + foot_contact_dim + local_map_dim + terrain_action_dim)
            except Exception:
                self.obs_dim = int(12 + 3 * self.num_actions)

            # finally set observation_space
            try:
                self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
            except Exception:
                self.observation_space = Box(low=-np.inf, high=np.inf, shape=(12 + 3 * self.num_actions,), dtype=np.float32)

        # prepare penalised / termination body ids according to rcfg if available
        self.penalised_body_ids = []
        self.termination_body_ids = []
        if self.rcfg is not None:
            penalize_list = getattr(self.rcfg.asset, 'penalize_contacts_on', [])
            terminate_list = getattr(self.rcfg.asset, 'terminate_after_contacts_on', [])
            for i in range(self.model.nbody):
                try:
                    name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i).decode() if isinstance(mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i), bytes) else mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
                except Exception:
                    name = ''
                lname = name.lower()
                for p in penalize_list:
                    if p.lower() in lname:
                        self.penalised_body_ids.append(int(i))
                for t in terminate_list:
                    if t.lower() in lname:
                        self.termination_body_ids.append(int(i))

        # find first hfield geom id (if any) to allow ground-relative height queries
        self.hfield_id = None
        self.hfield_geom_id = None
        try:
            # prefer the first hfield index if present
            if hasattr(self.model, 'hfield_nrow') and len(self.model.hfield_nrow) > 0:
                self.hfield_id = 0
            # find a geom whose name contains 'hfield' or 'terrain' to get center position
            for gi in range(self.model.ngeom):
                try:
                    gname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gi)
                    gname = gname.decode() if isinstance(gname, bytes) else gname
                except Exception:
                    gname = ''
                if gname and ('hfield' in gname.lower() or 'terrain' in gname.lower() or 'ground' in gname.lower()):
                    self.hfield_geom_id = gi
                    break
        except Exception:
            self.hfield_id = None
            self.hfield_geom_id = None

        # buffers used by several rewards
        self.last_contacts = np.zeros((len(self.foot_body_ids),), dtype=bool)
        self.feet_air_time = np.zeros((len(self.foot_body_ids),), dtype=np.float32)
        self.last_actions = np.zeros((self.num_actions,), dtype=np.float32)
        self.last_dof_vel = np.zeros((len(self.default_angles),), dtype=np.float32) if self.default_angles.size>0 else np.zeros((self.num_actions,), dtype=np.float32)

        # counters used to mirror TestEnv behaviour
        self.robot_counter = 0
        self.step_counter = 0
        # rendering flags (ControllerEnv typically headless)
        self.render = bool(self.cfg.get('visualization', {}).get('render', False))
        self.lock_camera = bool(self.cfg.get('visualization', {}).get('lock_camera', False))
        self.realtime_sim = bool(self.cfg.get('visualization', {}).get('realtime_sim', False))
        self.trace_enabled = bool(self.terrain_config.get('logging', {}).get('enable_trace', False))

        # termination statistics for debugging (counts per reason)
        self._termination_stats = {
            'fallen': 0,
            'collided': 0,
            'base_collision': 0,
            'thigh_collision': 0,
            'stuck': 0,
            'out_of_terrain_edge': 0,
            'time_limit': 0,
            'other': 0,
        }

        # optional debugging hook: when set to a numpy array, ControllerEnv will
        # use this tau vector each simulation step instead of computing PD.
        # This allows injecting the Trainer's tau to test whether PD/mapping
        # differences cause divergence.
        self.force_tau = None
        # when True, always use `force_tau` every muJoCo substep (skip internal PD)
        # Useful for debugging timing/PD mismatches — default False.
        self.force_trainer_tau_always = False

        # scale reward_scales by dt (per-step)
        try:
            self.dt = float(self.model.opt.timestep) * float(self.control_decimation)
            for k in list(self.reward_scales.keys()):
                if k != 'success':
                    # ensure numeric
                    try:
                        self.reward_scales[k] = float(self.reward_scales[k]) * self.dt
                    except Exception:
                        pass
        except Exception:
            self.dt = None

        # observation extras from terrain_config
        obs_cfg = self.terrain_config.get('observation', {})
        self.obs_include_foot_contacts = bool(obs_cfg.get('include_foot_contacts', False))
        self.obs_contact_force_threshold = float(obs_cfg.get('contact_force_threshold', 1.0))
        local_cfg = obs_cfg.get('local_height_map', {})
        self.local_map_enabled = bool(local_cfg.get('enabled', True))
        self.local_map_size_m = float(local_cfg.get('size_m', 1.0))
        self.local_map_resolution_m = float(local_cfg.get('resolution_m', 0.2))
        self.local_map_side = max(1, int(round(self.local_map_size_m / max(self.local_map_resolution_m, 1e-6))))

        # keep observation_space as controller input (robot_obs) only — model expects 48-dim
        try:
            robot_obs_dim = 12 + 3 * int(self.num_actions)
            self.obs_dim = int(robot_obs_dim)
            self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        except Exception:
            self.obs_dim = int(12 + 3 * int(self.num_actions))
            self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        mujoco.mj_resetData(self.model, self.data)
        self._step_count = 0
        # track simulated time since reset (include warmup)
        self._sim_time = 0.0
        # Match TestEnv: reset internal controller and terrain BEFORE warmup so the
        # warmup sees the same terrain and controller initial state.
        self._go2_controller.reset()

        # reset terrain before warmup to match TestEnv sequence
        if self.terrain_changer is not None:
            try:
                self.terrain_changer.reset(self.data)
            except Exception:
                self.terrain_changer.reset()

        # warmup initial steps with PD to settle — align with TestEnv init_skip_time + init_skip_frame
        target_dof_pos = self.default_angles.copy()
        try:
            total_sim_steps = int(self.init_skip_time / float(self.model.opt.timestep)) + int(self.init_skip_frame)
        except Exception:
            total_sim_steps = int(self.control_decimation * 10)
        for sim_i in range(total_sim_steps):
            # at control boundaries compute new target and tau; if we have an internal
            # Go2Controller use it to compute target to mirror TestEnv behaviour
            if sim_i >= 10 and sim_i % int(self.control_decimation) == 0:
                if self._go2_controller is not None:
                    try:
                        target_dof_pos = self._go2_controller.compute_action(self.data)
                    except Exception:
                        target_dof_pos = self.default_angles.copy()
                else:
                    target_dof_pos = self.default_angles.copy()
            # select gains from internal controller if available to match PD behaviour
            # prefer internal controller gains/scales when available
            kps = getattr(self._go2_controller, 'kps', self.kps)
            kds = getattr(self._go2_controller, 'kds', self.kds)
            # explicit joint slices
            try:
                num_j = int(self.num_actions)
            except Exception:
                num_j = len(kps) if hasattr(kps, '__len__') else 12
            qpos_j = self.data.qpos[7:7 + num_j]
            qvel_j = self.data.qvel[6:6 + num_j]
            # if forced trainer tau is present, optionally use it for every substep
            if getattr(self, 'force_trainer_tau_always', False) and getattr(self, 'force_tau', None) is not None:
                ft = np.asarray(self.force_tau, dtype=np.float32)
                tau = np.zeros_like(self.data.ctrl)
                tau[:min(ft.size, tau.size)] = ft[:min(ft.size, tau.size)]
            elif getattr(self, 'force_tau', None) is not None:
                # use provided full ctrl vector or slice to fill ctrl (single application)
                ft = np.asarray(self.force_tau, dtype=np.float32)
                tau = np.zeros_like(self.data.ctrl)
                tau[:min(ft.size, tau.size)] = ft[:min(ft.size, tau.size)]
            else:
                tau = pd_control(target_dof_pos, qpos_j, kps, np.zeros_like(kds), qvel_j, kds)
            # sanitize tau to avoid NaN/Inf or huge values destabilizing MuJoCo
            try:
                tau = np.nan_to_num(np.asarray(tau, dtype=np.float32), nan=0.0, posinf=1e6, neginf=-1e6)
                tau = np.clip(tau, -100.0, 100.0)
            except Exception:
                try:
                    tau = np.zeros_like(self.data.ctrl)
                except Exception:
                    tau = np.zeros((int(self.num_actions),), dtype=np.float32)
            self.data.ctrl[:] = tau
            mujoco.mj_step(self.model, self.data)
            self._sim_time += float(self.model.opt.timestep)
        # reset terrain changer hfield so controller env uses same terrain generation
        if self.terrain_changer is not None:
            try:
                self.terrain_changer.reset(self.data)
                # ensure terrain internal hfield/buffers are refreshed (matches TestEnv behaviour)
                try:
                    # TerrainChanger exposes a safe refresh helper in TestEnv
                    self.terrain_changer._refresh_terrain_safe()
                except Exception:
                    try:
                        # fallback to generic reset without args
                        self.terrain_changer.reset()
                    except Exception:
                        pass
            except Exception:
                try:
                    self.terrain_changer.reset()
                except Exception:
                    pass

        # Return (obs, info) to match Gymnasium-style reset used by VecEnv/Monitor
        return self._get_obs(), {}

    def _map_policy_action(self, action: np.ndarray) -> np.ndarray:
        """Map a policy-ordered action vector to the model joint order.

        If `self.policy2model` is present, it is an array of indices such that
        `action_model[self.policy2model] = action_policy`.
        Otherwise the input is returned unchanged.
        """
        action = np.asarray(action, dtype=np.float32)
        if getattr(self, 'policy2model', None) is None:
            return action
        try:
            model_action = np.empty_like(action)
            model_action[self.policy2model] = action
            return model_action
        except Exception:
            # fallback: return original
            return action

    def _compute_action_from_internal(self) -> np.ndarray:
        """Compute target_dof_pos by running the internal controller's compute_action logic.

        This inlines `Go2Controller.compute_action` behavior to avoid cross-module call
        differences: update command, build observation from `self.data`, run the
        policy network, update `action_policy_prev`, and return target_dof_pos.
        """
        ctrl = getattr(self, '_go2_controller', None)
        if ctrl is None:
            return self.default_angles.copy()
        try:
            # update command (may modify ctrl.cmd)
            try:
                cmd = getattr(ctrl, 'cmd', np.array(self.cfg.get('cmd_init', [0, 0, 0]), dtype=np.float32))
                ctrl.cmd = ctrl.update_command(self.data, cmd, getattr(ctrl, 'heading_stiffness', 0), getattr(ctrl, 'heading_target', 0), getattr(ctrl, 'heading_command', True))
            except Exception:
                pass

            # build observation exactly as Go2Controller.get_observation
            try:
                qj = self.data.qpos[7:].copy()
                dqj = self.data.qvel[6:].copy()
                quat = self.data.qpos[3:7].copy()
                lin_vel = self.data.qvel[:3].copy()
                ang_vel = self.data.qvel[3:6].copy()

                qj = (qj - ctrl.default_angles) * ctrl.dof_pos_scale
                dqj = dqj * ctrl.dof_vel_scale

                gravity_orientation = get_gravity_orientation(quat)
                lin_vel = lin_vel * ctrl.lin_vel_scale
                ang_vel = ang_vel * ctrl.ang_vel_scale

                obs = np.zeros(int(getattr(ctrl, 'num_obs', 48)), dtype=np.float32)
                obs[:3] = lin_vel
                obs[3:6] = ang_vel
                obs[6:9] = gravity_orientation
                obs[9:12] = ctrl.cmd * getattr(ctrl, 'cmd_scale', 1.0)
                obs[12: 12 + int(ctrl.num_actions)] = qj
                obs[12 + int(ctrl.num_actions): 12 + 2 * int(ctrl.num_actions)] = dqj
                obs[12 + 2 * int(ctrl.num_actions): 12 + 3 * int(ctrl.num_actions)] = ctrl.action_policy_prev
            except Exception:
                obs = np.zeros(int(getattr(ctrl, 'num_obs', 48)), dtype=np.float32)

            # policy inference
            try:
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                action_policy = ctrl.policy(obs_tensor).detach().cpu().numpy().squeeze()
            except Exception:
                action_policy = np.zeros(int(getattr(ctrl, 'num_actions', self.num_actions)), dtype=np.float32)

            # model action order conversion - prefer the controller's configured
            # action_scale/default_angles to avoid mismatches between modules.
            try:
                a_scale = getattr(ctrl, 'action_scale', self.action_scale)
                d_angles = getattr(ctrl, 'default_angles', self.default_angles)
                target_dof_pos = action_policy * a_scale + d_angles
            except Exception:
                target_dof_pos = getattr(ctrl, 'default_angles', self.default_angles).copy()

            # save for next observation
            try:
                ctrl.action_policy_prev[:] = action_policy
            except Exception:
                pass

            return target_dof_pos
        except Exception:
            return self.default_angles.copy()

    def _safe_call(self, func, *args, **kwargs):
        # helper similar to TestEnv.safe_call to accept optional kwargs
        import inspect
        try:
            sig = inspect.signature(func)
            valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
            return func(*args, **valid_kwargs)
        except Exception:
            return func(*args)

    # ---------- helpers to align obs with TestEnv ----------
    def _get_foot_contact_flags(self):
        flags = np.zeros((len(self.foot_body_ids),), dtype=np.float32)
        if len(self.foot_body_ids) == 0:
            return flags

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, i, force)
            if np.linalg.norm(force[:3]) < self.obs_contact_force_threshold:
                continue
            b1 = int(self.model.geom_bodyid[contact.geom1])
            b2 = int(self.model.geom_bodyid[contact.geom2])
            for j, fb in enumerate(self.foot_body_ids):
                if b1 == fb or b2 == fb:
                    flags[j] = 1.0
        return flags

    def _get_local_height_map_obs(self):
        if not self.local_map_enabled or self.terrain_changer is None:
            return np.zeros((0,), dtype=np.float32)

        cx = float(self.data.qpos[0])
        cy = float(self.data.qpos[1])
        center_h = self._get_ground_height_at_xy(cx, cy)

        n = self.local_map_side
        res = self.local_map_resolution_m
        start = -0.5 * self.local_map_size_m + 0.5 * res
        end = 0.5 * self.local_map_size_m - 0.5 * res
        xs = np.linspace(start, end, n, dtype=np.float32)
        ys = np.linspace(start, end, n, dtype=np.float32)

        vals = np.zeros((n, n), dtype=np.float32)
        for iy, oy in enumerate(ys):
            for ix, ox in enumerate(xs):
                h = self._get_ground_height_at_xy(cx + float(ox), cy + float(oy))
                vals[iy, ix] = float(h - center_h)
        return vals.reshape(-1)

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
        # if policy uses a different joint ordering, map to model order
        try:
            action_model = self._map_policy_action(action)
        except Exception:
            action_model = action
        # keep Go2Controller internal state (cmd, previous action) in sync with external action
        if getattr(self, '_go2_controller', None) is not None:
            try:
                # update heading/velocity command using controller helper (matches compute_action)
                self._go2_controller.cmd = self._go2_controller.update_command(self.data,
                                                                            getattr(self._go2_controller, 'cmd', np.array(self.cfg.get('cmd_init', [0,0,0]), dtype=np.float32)),
                                                                            getattr(self._go2_controller, 'heading_stiffness', getattr(self._go2_controller, 'heading_stiffness', 0)),
                                                                            getattr(self._go2_controller, 'heading_target', getattr(self._go2_controller, 'heading_target', 0)),
                                                                            getattr(self._go2_controller, 'heading_command', getattr(self._go2_controller, 'heading_command', True)))
            except Exception:
                pass
            try:
                # record the last policy action as the controller expects
                # controller expects previous action in policy order
                self._go2_controller.action_policy_prev[:] = action
            except Exception:
                pass

        # compute target dof pos in model order (prefer internal controller's params)
        if getattr(self, '_go2_controller', None) is not None:
            ctrl = self._go2_controller
            a_scale = getattr(ctrl, 'action_scale', self.action_scale)
            d_angles = getattr(ctrl, 'default_angles', self.default_angles)
            target_dof_pos = action_model * a_scale + d_angles
        else:
            target_dof_pos = action_model * self.action_scale + self.default_angles

        # run robot controller for a full terrain step (terrain_decimation * control_decimation sim steps)
        total_sim_steps = int(getattr(self, 'terrain_decimation', 1)) * int(self.control_decimation)
        # target_dof_pos already computed in model order above (use action_model)
        # prefer controller-provided gains to match TestEnv exactly
        kps = getattr(self._go2_controller, 'kps', self.kps)
        kds = getattr(self._go2_controller, 'kds', self.kds)

        for sim_i in range(total_sim_steps):
            self.robot_counter += 1
            step_start = time.time()
            # at control boundaries compute new target and tau (zero-order-hold on policy action)
            if sim_i % int(self.control_decimation) == 0:
                # action held constant across the whole terrain step; use the incoming
                # policy action (mapped to model order) rather than recomputing from
                # the internal controller. This ensures the step() input controls
                # the robot state as intended.
                try:
                    if getattr(self, '_go2_controller', None) is not None:
                        ctrl = self._go2_controller
                        a_scale = getattr(ctrl, 'action_scale', self.action_scale)
                        d_angles = getattr(ctrl, 'default_angles', self.default_angles)
                        target_dof_pos = action_model * a_scale + d_angles
                    else:
                        target_dof_pos = action_model * self.action_scale + self.default_angles
                except Exception:
                    target_dof_pos = action_model * self.action_scale + self.default_angles
            # explicit joint slices
            try:
                num_j = int(self.num_actions)
            except Exception:
                num_j = len(kps) if hasattr(kps, '__len__') else 12
            qpos_j = self.data.qpos[7:7 + num_j]
            qvel_j = self.data.qvel[6:6 + num_j]
            # optional per-sim-step trace collection enabled via harness attributes
            collect_sim_trace = getattr(self, 'simtrace_enable', False) and getattr(self, 'simtrace_out_dir', None) is not None
            if getattr(self, 'force_trainer_tau_always', False) and getattr(self, 'force_tau', None) is not None:
                ft = np.asarray(self.force_tau, dtype=np.float32)
                tau = np.zeros_like(self.data.ctrl)
                tau[:min(ft.size, tau.size)] = ft[:min(ft.size, tau.size)]
            elif getattr(self, 'force_tau', None) is not None:
                ft = np.asarray(self.force_tau, dtype=np.float32)
                tau = np.zeros_like(self.data.ctrl)
                tau[:min(ft.size, tau.size)] = ft[:min(ft.size, tau.size)]
            else:
                tau = pd_control(target_dof_pos, qpos_j, kps, np.zeros_like(kds), qvel_j, kds)
                # sanitize tau to avoid NaN/Inf or huge values destabilizing MuJoCo
                try:
                    tau = np.nan_to_num(np.asarray(tau, dtype=np.float32), nan=0.0, posinf=1e6, neginf=-1e6)
                    tau = np.clip(tau, -100.0, 100.0)
                except Exception:
                    try:
                        tau = np.zeros_like(self.data.ctrl)
                    except Exception:
                        tau = np.zeros((int(self.num_actions),), dtype=np.float32)
            self.data.ctrl[:] = tau
            # record snapshot before stepping (helps compare per-mj-step traces)
            if collect_sim_trace:
                try:
                    if not hasattr(self, '_sim_trace_buffer'):
                        self._sim_trace_buffer = []
                    self._sim_trace_buffer.append({
                        'sim_i': int(sim_i),
                        'qpos': np.asarray(self.data.qpos).tolist(),
                        'qvel': np.asarray(self.data.qvel).tolist(),
                        'tau': np.asarray(tau).tolist(),
                        'ctrl': np.asarray(self.data.ctrl).tolist(),
                    })
                except Exception:
                    pass
            mujoco.mj_step(self.model, self.data)

            if self.render:
                try:
                    if self.lock_camera:
                        # viewer may not exist; guard it
                        try:
                            self.viewer.cam.lookat[:] = self.data.qpos[:3]
                        except Exception:
                            pass
                    try:
                        self.viewer.sync()
                    except Exception:
                        pass
                except Exception:
                    pass

        self._step_count += 1
        self.step_counter += 1

        # flush sim-trace buffer to disk if enabled and within configured limit
        try:
            if getattr(self, 'simtrace_enable', False) and getattr(self, 'simtrace_out_dir', None) is not None:
                try:
                    ep = int(getattr(self, 'simtrace_ep', 0))
                except Exception:
                    ep = 0
                try:
                    step_idx = int(self.step_counter)
                except Exception:
                    step_idx = 0
                try:
                    limit = int(getattr(self, 'simtrace_limit_steps', 3))
                except Exception:
                    limit = 3
                if step_idx <= limit:
                    try:
                        import json as _json, os as _os
                        _os.makedirs(self.simtrace_out_dir, exist_ok=True)
                        _path = _os.path.join(self.simtrace_out_dir, f"controller_sim_ep{ep}_step{step_idx}.json")
                        buf = getattr(self, '_sim_trace_buffer', None)
                        if not buf:
                            # no buffer collected (fallback): write a single snapshot of current state
                            try:
                                buf = [{
                                    'sim_i': -1,
                                    'qpos': np.asarray(self.data.qpos).tolist(),
                                    'qvel': np.asarray(self.data.qvel).tolist(),
                                    'tau': np.asarray(self.data.ctrl).tolist(),
                                    'ctrl': np.asarray(self.data.ctrl).tolist(),
                                }]
                            except Exception:
                                buf = []
                        with open(_path, 'w') as _f:
                            _json.dump(buf, _f)
                    except Exception:
                        pass
                try:
                    delattr(self, '_sim_trace_buffer')
                except Exception:
                    try:
                        del self._sim_trace_buffer
                    except Exception:
                        pass
        except Exception:
            pass

        # mirror TestEnv: update terrain_changer.last_action if terrain_changer exists
        try:
            if hasattr(self, 'terrain_changer') and getattr(self.terrain_changer, 'last_action', None) is None:
                # if last_action not set, attempt to set from internal state or zeros
                try:
                    self.terrain_changer.last_action = np.zeros(int(getattr(self, 'terrain_action_dim', 0)), dtype=np.float32)
                except Exception:
                    try:
                        self.terrain_changer.last_action = np.zeros((1,), dtype=np.float32)
                    except Exception:
                        pass
        except Exception:
            pass
        # advance simulated time by the number of sim steps taken
        try:
            self._sim_time += float(total_sim_steps) * float(self.model.opt.timestep)
        except Exception:
            pass

        obs = self._get_obs()
        reward, info, done = self._compute_reward_and_info(action)

        # termination/truncation bookkeeping
        truncated = False
        if self._step_count >= self.max_episode_steps:
            truncated = True
            done = True

        # final done also triggers if any fail flags present in info
        final_done = bool(done) or bool(info.get('fallen', False) or info.get('collided', False) or info.get('base_collision', False) or info.get('thigh_collision', False) or info.get('stuck', False) or info.get('out_of_terrain_edge', False))

        # determine if this termination is a failure (any failure flag)
        failed = bool(info.get('fallen', False) or info.get('collided', False) or info.get('base_collision', False) or info.get('thigh_collision', False) or info.get('stuck', False))

        # give a positive success reward when episode ended (done) and not a failure
        if final_done and (not failed):
            info['success'] = True
            # prefer configured per-step reward_scales['success'] if present, else use default episodic reward
            success_scale = float(self.reward_scales.get('success', 1.0))
            reward += success_scale
        else:
            info['success'] = False

        # record termination statistics for debugging
        if final_done:
            reason = 'other'
            if bool(info.get('fallen', False)):
                reason = 'fallen'
            elif bool(info.get('collided', False)):
                reason = 'collided'
            elif bool(info.get('base_collision', False)):
                reason = 'base_collision'
            elif bool(info.get('thigh_collision', False)):
                reason = 'thigh_collision'
            elif bool(info.get('stuck', False)):
                reason = 'stuck'
            elif bool(info.get('out_of_terrain_edge', False)):
                reason = 'out_of_terrain_edge'
            elif truncated:
                reason = 'time_limit'
            try:
                self._termination_stats.setdefault(reason, 0)
                self._termination_stats[reason] += 1
            except Exception:
                pass

        # Return Gymnasium-compatible 5-tuple (obs, reward, terminated, truncated, info)
        terminated = bool(final_done)
        truncated = bool(truncated)
        return obs, float(reward), terminated, truncated, info

    def _get_obs(self):
        # compose observation similar to Go2Controller.get_observation
        qj = self.data.qpos[7:].copy()
        dqj = self.data.qvel[6:].copy()
        quat = self.data.qpos[3:7].copy()
        lin_vel = self.data.qvel[:3].copy()
        ang_vel = self.data.qvel[3:6].copy()

        # prefer internal controller's scaling/offsets when available
        if getattr(self, '_go2_controller', None) is not None:
            ctrl = self._go2_controller
            _def_angles = getattr(ctrl, 'default_angles', self.default_angles)
            _dof_pos_scale = getattr(ctrl, 'dof_pos_scale', self.dof_pos_scale)
            _dof_vel_scale = getattr(ctrl, 'dof_vel_scale', self.dof_vel_scale)
        else:
            _def_angles = self.default_angles
            _dof_pos_scale = self.dof_pos_scale
            _dof_vel_scale = self.dof_vel_scale

        qj = (qj - _def_angles) * _dof_pos_scale
        dqj = dqj * _dof_vel_scale

        gravity_orientation = get_gravity_orientation(quat)
        lin_vel = lin_vel * self.lin_vel_scale
        ang_vel = ang_vel * self.cfg.get("ang_vel_scale", 1.0)

        robot_obs = np.zeros(self.robot_obs_dim, dtype=np.float32)
        robot_obs[:3] = lin_vel
        robot_obs[3:6] = ang_vel
        robot_obs[6:9] = gravity_orientation
        # cmd placeholder
        # include command scaled similarly to Go2Controller
        if getattr(self, '_go2_controller', None) is not None:
            ctrl_cmd = getattr(self._go2_controller, 'cmd', np.array(self.cfg.get("cmd_init", [0,0,0]), dtype=np.float32))
            ctrl_cmd_scale = getattr(self._go2_controller, 'cmd_scale', self.cfg.get('cmd_scale', 1.0))
            # normalize to numpy arrays and broadcast to length 3
            ctrl_cmd = np.asarray(ctrl_cmd, dtype=np.float32)
            ctrl_cmd_scale = np.asarray(ctrl_cmd_scale, dtype=np.float32)
            # ensure ctrl_cmd_scale has at least 3 elements (broadcast single value)
            if ctrl_cmd_scale.size == 1:
                ctrl_cmd_scale = np.repeat(ctrl_cmd_scale, 3)
            else:
                if ctrl_cmd_scale.size < 3:
                    pad = np.ones((3 - ctrl_cmd_scale.size,), dtype=np.float32)
                    ctrl_cmd_scale = np.concatenate([ctrl_cmd_scale, pad], axis=0)
            # apply elementwise scaling and take first 3 values
            scaled_cmd = (ctrl_cmd * ctrl_cmd_scale)[:3]
            # pad/truncate to exactly 3
            if scaled_cmd.size < 3:
                tmp = np.zeros((3,), dtype=np.float32)
                tmp[:scaled_cmd.size] = scaled_cmd
                scaled_cmd = tmp
            robot_obs[9:12] = scaled_cmd
        else:
            robot_obs[9:12] = np.array(self.cfg.get("cmd_init", [0,0,0]), dtype=np.float32)
        robot_obs[12: 12 + self.num_actions] = qj
        # no per-sim-step tracing from within _get_obs (traces are collected in step())
        robot_obs[12 + self.num_actions: 12 + 2 * self.num_actions] = dqj
        robot_obs[12 + 2 * self.num_actions: 12 + 3 * self.num_actions] = self.last_actions

        # Return robot-only observation (controller input). Extra terrain/contacts are available
        # via other helpers but are not part of the policy input to keep model shape = 48.
        return robot_obs

    # ----- terrain helpers (partial port of TerrainTrainer helpers) -----
    def _get_ground_height_at_xy(self, x: float, y: float) -> float:
        """Estimate terrain surface height at world XY from model hfield if available."""
        try:
            # Prefer terrain_changer's hfield if available (matches TestEnv behaviour)
            if hasattr(self, 'terrain_changer') and getattr(self.terrain_changer, 'hfield', None) is not None:
                try:
                    hfield = self.terrain_changer.hfield
                    geom_id = int(getattr(self.terrain_changer, 'geom_id', getattr(self.terrain_changer, 'hfield_geom_id', None)))
                    if geom_id is None:
                        return 0.0
                    nrow = int(hfield.shape[0])
                    ncol = int(hfield.shape[1])
                    size_x = float(getattr(self.terrain_changer, 'terrain_size_x', float(self.model.hfield_size[self.hfield_id][0]) if self.hfield_id is not None else 0.5))
                    size_y = float(getattr(self.terrain_changer, 'terrain_size_y', float(self.model.hfield_size[self.hfield_id][1]) if self.hfield_id is not None else 0.5))
                    z_scale = float(getattr(self.terrain_changer, 'terrain_z_scale', 1.0))

                    center_x = float(getattr(self.terrain_changer, 'terrain_center_x', float(self.model.geom_pos[geom_id][0])))
                    center_y = float(getattr(self.terrain_changer, 'terrain_center_y', float(self.model.geom_pos[geom_id][1])))
                    center_z = float(getattr(self.terrain_changer, 'terrain_center_z', float(self.model.geom_pos[geom_id][2])))

                    u = (x - center_x + size_x) / (2.0 * size_x)
                    v = (y - center_y + size_y) / (2.0 * size_y)
                    u = float(np.clip(u, 0.0, 1.0))
                    v = float(np.clip(v, 0.0, 1.0))

                    col = int(np.clip(round(u * (ncol - 1)), 0, ncol - 1))
                    row = int(np.clip(round(v * (nrow - 1)), 0, nrow - 1))

                    h = float(hfield[row, col])
                    return center_z + z_scale * h
                except Exception:
                    return 0.0

            # Fallback to model hfield if present
            if self.hfield_id is None or self.hfield_geom_id is None:
                return 0.0
            hfield_id = int(self.hfield_id)
            nrow = int(self.model.hfield_nrow[hfield_id])
            ncol = int(self.model.hfield_ncol[hfield_id])
            size_x = float(self.model.hfield_size[hfield_id][0])
            size_y = float(self.model.hfield_size[hfield_id][1])
            z_scale = float(self.model.hfield_size[hfield_id][2])

            center_x = float(self.model.geom_pos[self.hfield_geom_id][0])
            center_y = float(self.model.geom_pos[self.hfield_geom_id][1])
            center_z = float(self.model.geom_pos[self.hfield_geom_id][2])

            u = (x - center_x + size_x) / (2.0 * size_x)
            v = (y - center_y + size_y) / (2.0 * size_y)
            u = float(np.clip(u, 0.0, 1.0))
            v = float(np.clip(v, 0.0, 1.0))

            col = int(np.clip(round(u * (ncol - 1)), 0, ncol - 1))
            row = int(np.clip(round(v * (nrow - 1)), 0, nrow - 1))

            try:
                h = float(self.model.hfield_data[row * ncol + col])
            except Exception:
                h = 0.0
            return center_z + z_scale * h
        except Exception:
            return 0.0

    def _is_fallen_ground_relative(self, base_z: float, roll: float, pitch: float) -> bool:
        # ground-relative check using terrain_config thresholds
        try:
            x = float(self.data.qpos[0])
            y = float(self.data.qpos[1])
            ground_z = self._get_ground_height_at_xy(x, y)
            base_rel = base_z - ground_z
            if base_rel < float(self._fall_height_threshold):
                return True
            return abs(roll) > float(self._fall_angle_threshold) or abs(pitch) > float(self._fall_angle_threshold)
        except Exception:
            return False

    def _analyze_contacts(self):
        # classify collisions into collided/base_collision/thigh_collision using terrain thresholds
        collided = False
        base_collision = False
        thigh_collision = False
        force_thresh = float(self._collision_force_threshold)

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, i, force)
            f_mag = np.linalg.norm(force[:3])
            if f_mag < force_thresh:
                continue
            collided = True
            try:
                geom1 = contact.geom1
                geom2 = contact.geom2
                body1 = int(self.model.geom_bodyid[geom1])
                body2 = int(self.model.geom_bodyid[geom2])
                name1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body1)
                name1 = name1.decode() if isinstance(name1, bytes) else name1
                name2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body2)
                name2 = name2.decode() if isinstance(name2, bytes) else name2
            except Exception:
                name1 = ''
                name2 = ''
            if 'base' in (name1 or '').lower() or 'base' in (name2 or '').lower():
                base_collision = True
            if 'thigh' in (name1 or '').lower() or 'thigh' in (name2 or '').lower():
                thigh_collision = True

        # apply failure flags
        flags = self._failure_flags if isinstance(self._failure_flags, dict) else {}
        collided = collided and bool(flags.get('collided', True))
        base_collision = base_collision and bool(flags.get('base_collision', True))
        thigh_collision = thigh_collision and bool(flags.get('thigh_collision', True))

        return collided, base_collision, thigh_collision

    def _is_out_of_terrain_edge(self) -> bool:
        try:
            # if no hfield geom info, cannot compute -> False
            if self.hfield_id is None or self.hfield_geom_id is None:
                return False
            size_x = float(self.model.hfield_size[self.hfield_id][0]) * 2.0
            size_y = float(self.model.hfield_size[self.hfield_id][1]) * 2.0
            center_x = float(self.model.geom_pos[self.hfield_geom_id][0])
            center_y = float(self.model.geom_pos[self.hfield_geom_id][1])
            margin = float(self._termination_cfg.get('terrain_edge_margin', 1.0))
            half_x = size_x * 0.5 - margin
            half_y = size_y * 0.5 - margin
            if half_x <= 0.0 or half_y <= 0.0:
                return False
            x = float(self.data.qpos[0])
            y = float(self.data.qpos[1])
            return (abs(x - center_x) >= half_x) or (abs(y - center_y) >= half_y)
        except Exception:
            return False

    def _is_stuck(self, lin_vel: float, target_speed: float) -> Tuple[bool, float]:
        stuck = lin_vel < float(self._stuck_speed_threshold) and target_speed > 0.2
        return stuck, float(self.terrain_config.get('event_and_reward', {}).get('stuck_reward', 2.0)) if stuck else (False, 0.0)

    def _compute_tilt_reward(self, roll: float, pitch: float) -> Tuple[float, float]:
        """Return (tilt, tilt_reward) similar to TestEnv."""
        tilt = abs(roll) + abs(pitch)
        val = float(self.terrain_config.get('event_and_reward', {}).get('tilt_reward_scale', 0.0)) * tilt
        return float(tilt), float(val)

    def _compute_speed_reward(self, lin_vel: float, target_speed: float) -> float:
        """Return speed loss scaled reward (matching TestEnv)."""
        speed_loss = max(0.0, target_speed - lin_vel)
        return float(self.terrain_config.get('event_and_reward', {}).get('speed_reward_scale', 0.0)) * speed_loss

    def _compute_reward_and_info(self, action):
        # Compute controller-style reward components similarly to TestEnv.compute_terrain_reward
        try:
            base_z = float(self.data.qpos[2])
            lin_vel = float(np.linalg.norm(self.data.qvel[:2]))
            quat = self.data.qpos[3:7]
            roll, pitch, _ = quat_to_rpy(quat)

            # analyze contacts
            collided, base_collision, thigh_collision = self._analyze_contacts()

            # termination contact: any contact involving termination bodies with >1.0 force
            termination_contact = False
            try:
                for i in range(self.data.ncon):
                    contact = self.data.contact[i]
                    force = np.zeros(6)
                    mujoco.mj_contactForce(self.model, self.data, i, force)
                    f_mag = np.linalg.norm(force[:3])
                    try:
                        b1 = int(self.model.geom_bodyid[contact.geom1])
                        b2 = int(self.model.geom_bodyid[contact.geom2])
                    except Exception:
                        b1 = -1
                        b2 = -1
                    if (b1 in self.termination_body_ids or b2 in self.termination_body_ids) and f_mag > 1.0:
                        termination_contact = True
                        break
            except Exception:
                termination_contact = False

            # fallen check
            fallen = self._is_fallen_ground_relative(base_z, roll, pitch)

            reward = 0.0
            comp_rewards = {}

            # If controller reward scales are available, use them (same logic as TestEnv)
            if self.reward_scales:
                # tracking linear velocity
                if 'tracking_lin_vel' in self.reward_scales:
                    cmd = np.array(self.cfg.get('cmd_init', [1.0, 0, 0]), dtype=np.float32)
                    lin_err = np.sum((cmd[:2] - self.data.qvel[:2]) ** 2)
                    tracking_lin = np.exp(- lin_err / max(1e-6, self.tracking_sigma))
                    val = float(self.reward_scales.get('tracking_lin_vel', 0.0) * tracking_lin)
                    reward += val
                    comp_rewards['tracking_lin_vel'] = val

                # tracking angular vel (yaw)
                if 'tracking_ang_vel' in self.reward_scales:
                    cmd = np.array(self.cfg.get('cmd_init', [0, 0, 0]), dtype=np.float32)
                    ang_err = (cmd[2] - float(self.data.qvel[5])) ** 2
                    tracking_ang = np.exp(- ang_err / max(1e-6, self.tracking_sigma))
                    val = float(self.reward_scales.get('tracking_ang_vel', 0.0) * tracking_ang)
                    reward += val
                    comp_rewards['tracking_ang_vel'] = val

                # torque penalty (approximate via PD call)
                if 'torques' in self.reward_scales:
                    try:
                        target_dof_pos = action * self.action_scale + self.default_angles
                        # use controller gains if available to match TestEnv torque calculation
                        _kps = getattr(self._go2_controller, 'kps', self.kps)
                        _kds = getattr(self._go2_controller, 'kds', self.kds)
                        try:
                            num_j = int(getattr(self, 'num_actions', len(_kps) if hasattr(_kps, '__len__') else 12))
                        except Exception:
                            num_j = len(_kps) if hasattr(_kps, '__len__') else 12
                        qpos_j = self.data.qpos[7:7 + num_j]
                        qvel_j = self.data.qvel[6:6 + num_j]
                        tau = pd_control(target_dof_pos, qpos_j, _kps, np.zeros_like(_kds), qvel_j, _kds)
                        torque_pen = np.sum(np.square(tau))
                        val = float(self.reward_scales.get('torques', 0.0) * torque_pen)
                        reward += val
                        comp_rewards['torques'] = val
                    except Exception:
                        comp_rewards['torques'] = 0.0

                # dof pos limits penalty
                if 'dof_pos_limits' in self.reward_scales:
                    dof_pos = np.asarray(self.data.qpos[7:])
                    soft_limit = 1.0
                    try:
                        soft_limit = float(getattr(self.rcfg.rewards, 'soft_dof_pos_limit', 1.0)) if self.rcfg is not None else 1.0
                    except Exception:
                        soft_limit = 1.0
                    limit = 2.0 * soft_limit
                    out_limits = np.sum(np.maximum(0.0, np.abs(dof_pos) - limit))
                    val = float(self.reward_scales.get('dof_pos_limits', 0.0) * out_limits)
                    reward += val
                    comp_rewards['dof_pos_limits'] = val

                # collision penalty
                if 'collision' in self.reward_scales and collided:
                    val = float(self.reward_scales.get('collision', 0.0))
                    reward += val
                    comp_rewards['collision'] = val
                else:
                    comp_rewards.setdefault('collision', 0.0)

                # stuck penalty/reward
                stuck, stuck_reward = self._is_stuck(lin_vel, float(self.terrain_config.get('event_and_reward', {}).get('target_speed', 1.0)))
                if stuck:
                    val = float(self.reward_scales.get('stuck', stuck_reward)) if 'stuck' in self.reward_scales else float(stuck_reward)
                    reward += val
                    comp_rewards['stuck'] = val
                else:
                    comp_rewards['stuck'] = 0.0

                # termination reward
                if 'termination' in self.reward_scales and (fallen or termination_contact):
                    val = float(self.reward_scales.get('termination', 0.0))
                    reward += val
                    comp_rewards['termination'] = val
                else:
                    comp_rewards.setdefault('termination', 0.0)

                # apply only_positive clipping
                if getattr(self, 'only_positive_rewards', False):
                    reward = max(0.0, float(reward))

                # ensure common components (tilt/speed)
                tilt_val, tilt_reward_val = self._compute_tilt_reward(roll, pitch)
                comp_rewards.setdefault('tilt', float(tilt_reward_val))
                comp_rewards.setdefault('speed', float(self._compute_speed_reward(lin_vel, float(self.terrain_config.get('event_and_reward', {}).get('target_speed', 1.0)))))

            else:
                # fallback to original simplistic accumulation
                reward = 0.0
                comp_rewards = {}
                # fallen
                repeat = bool(self.terrain_config.get('event_and_reward', {}).get('repeat_reward', False))
                if fallen:
                    val = float(self.terrain_config.get('event_and_reward', {}).get('fall_reward', 0.0)) if (repeat or not getattr(self, '_fallen_reported', False)) else 0.0
                    reward += val
                    comp_rewards['fallen'] = val
                # collisions
                col_val = 0.0
                if collided:
                    col_val = float(self.terrain_config.get('event_and_reward', {}).get('collision_reward', 0.0))
                reward += col_val
                comp_rewards['collision'] = float(col_val)

            # update feet air time/contact based rewards (preserve previous behaviour)
            curr_contacts = []
            for fb in self.foot_body_ids:
                has_contact = False
                for i in range(self.data.ncon):
                    contact = self.data.contact[i]
                    b1 = int(self.model.geom_bodyid[contact.geom1])
                    b2 = int(self.model.geom_bodyid[contact.geom2])
                    if b1 == fb or b2 == fb:
                        force = np.zeros(6)
                        mujoco.mj_contactForce(self.model, self.data, i, force)
                        if np.linalg.norm(force[:3]) > 1.0:
                            has_contact = True
                            break
                curr_contacts.append(has_contact)

            for i, contacted in enumerate(curr_contacts):
                if not contacted:
                    self.feet_air_time[i] += self.model.opt.timestep * self.control_decimation
                else:
                    if self.feet_air_time[i] > 0.5:
                        val = float(self.reward_scales.get('feet_air_time', 0.0) * (self.feet_air_time[i] - 0.5)) if self.reward_scales else 0.0
                        reward += val
                        comp_rewards['feet_air_time'] = comp_rewards.get('feet_air_time', 0.0) + val
                    self.feet_air_time[i] = 0.0

            # update last buffers
            self.last_actions = np.asarray(action, dtype=np.float32)
            self.last_dof_vel = np.asarray(self.data.qvel[6:], dtype=np.float32)

            # compute out_of_terrain_edge and stuck (and possibly ignore stuck during warmup)
            out_of_terrain_edge = self._is_out_of_terrain_edge()
            target_speed = float(self.terrain_config.get('event_and_reward', {}).get('target_speed', 1.0))
            try:
                ignore_stuck = (hasattr(self, '_sim_time') and self._sim_time < float(self.init_skip_time))
            except Exception:
                ignore_stuck = False
            if ignore_stuck:
                stuck = False
            else:
                stuck, _ = self._is_stuck(lin_vel, target_speed)

            # assemble info dict with rewards breakdown (matching TestEnv)
            x = float(self.data.qpos[0])
            y = float(self.data.qpos[1])
            ground_height = self._get_ground_height_at_xy(x, y)
            base_rel_height = base_z - ground_height

            info = {
                'fallen': fallen,
                'collided': collided,
                'base_collision': base_collision,
                'thigh_collision': thigh_collision,
                'stuck': stuck,
                'out_of_terrain_edge': out_of_terrain_edge,
                'speed': lin_vel,
                'base_height': base_z,
                'ground_height': ground_height,
                'base_rel_height': base_rel_height,
                'rewards': comp_rewards,
            }

            # termination decision
            done = False
            if self._failure_flags and self._failure_flags.get('fallen', True) and fallen and self._termination_cfg.get('terminate_on_fall', False):
                done = True
            if self._failure_flags and self._failure_flags.get('base_collision', True) and info.get('base_collision', False) and self._termination_cfg.get('terminate_on_base_collision', False):
                done = True
            # Match TestEnv: only treat 'stuck' as terminal when accompanied by base collision
            if self._failure_flags and self._failure_flags.get('stuck', True) and info.get('stuck', False) and info.get('base_collision', False) and self._termination_cfg.get('terminate_on_stuck', False):
                done = True
            if out_of_terrain_edge and self._termination_cfg.get('terminate_on_terrain_edge', False):
                done = True

            return float(reward), info, bool(done)
        except Exception:
            # on error, fallback to earlier simple behaviour
            try:
                base_z = float(self.data.qpos[2])
                lin_vel = float(np.linalg.norm(self.data.qvel[:2]))
                quat = self.data.qpos[3:7]
                roll, pitch, _ = quat_to_rpy(quat)
                fallen = self._is_fallen_ground_relative(base_z, roll, pitch)
                collided, base_collision, thigh_collision = self._analyze_contacts()
                reward = 0.0
                repeat = bool(self.terrain_config.get('event_and_reward', {}).get('repeat_reward', False))
                # basic fallback components
                if fallen:
                    reward += float(self.terrain_config.get('event_and_reward', {}).get('fall_reward', 0.0))
                reward += float(self.terrain_config.get('event_and_reward', {}).get('collision_reward', 0.0)) if collided else 0.0
                tilt, tilt_reward = self._compute_tilt_reward(roll, pitch)
                reward += tilt_reward
                reward += self._compute_speed_reward(lin_vel, float(self.terrain_config.get('event_and_reward', {}).get('target_speed', 1.0)))
                stuck, stuck_reward = self._is_stuck(lin_vel, float(self.terrain_config.get('event_and_reward', {}).get('target_speed', 1.0)))
                reward += stuck_reward
                out_of_terrain_edge = self._is_out_of_terrain_edge()
                done = False
                if fallen and self._termination_cfg.get('terminate_on_fall', False):
                    done = True
                info = {
                    'fallen': fallen,
                    'collided': collided,
                    'base_collision': base_collision,
                    'thigh_collision': thigh_collision,
                    'stuck': stuck,
                    'out_of_terrain_edge': out_of_terrain_edge,
                    'speed': lin_vel,
                    'base_height': base_z,
                    'rewards': {},
                }
                return float(reward), info, bool(done)
            except Exception:
                return 0.0, {}, False

    def get_termination_stats(self):
        """Return a copy of accumulated termination statistics."""
        try:
            return dict(self._termination_stats)
        except Exception:
            return {}
