import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

import mujoco
import numpy as np
import yaml
from typing import Tuple
try:
    import gymnasium as gym
    from gymnasium.spaces import Box
except ImportError:
    import gym
    from gym.spaces import Box
from deploy_mujoco.utils import quat_to_rpy, pd_control
import mujoco.viewer
import time
from deploy_mujoco.terrain_params import TerrainChanger
import inspect


def safe_call(func, *args, **kwargs):
    sig = inspect.signature(func)
    valid_kwargs = {
        k: v for k, v in kwargs.items()
        if k in sig.parameters
    }
    return func(*args, **valid_kwargs)


class TerrainTrainer:
    """Environment wrapper that runs the Go2 controller at a high frequency and
    exposes a lower-frequency API for a terrain-controlling agent.

    This class currently supports bump (implemented) and provides interfaces for
    slide_friction and solref (placeholders that call through to the TerrainChanger).
    """

    def __init__(
        self,
        go2_config_file,
        terrain_config_file,
    ):

        if go2_config_file[0] == "terrain":
            from deploy_mujoco.terrain.go2_controller import Go2Controller
        else:
            from deploy_mujoco.terrain.go2_controller import Go2Controller

        self.go2_controller = Go2Controller(go2_config_file[1])

        # Load Go2 controller config
        with open(f"{os.path.dirname(os.path.realpath(__file__))}/{go2_config_file[0]}/configs/{go2_config_file[1]}", "r", encoding='utf-8') as f:
            self.go2_config = yaml.load(f, Loader=yaml.FullLoader)

        # MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(self.go2_config["xml_path"])
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = self.go2_config["simulation_dt"]  # 仿真步长，等于Go2的控制步长

        # Terrain setup
        with open(f"{os.path.dirname(os.path.realpath(__file__))}/{terrain_config_file}", "r", encoding="utf-8") as f:
            self.terrain_config = yaml.load(f, Loader=yaml.FullLoader)
        self.terrain_decimation = self.terrain_config.get("terrain_action", {}).get("terrain_decimation", 0)
        self.terrain_types = self.terrain_config.get("terrain_action", {}).get("terrain_types", [])
        self.failure_flags = self.terrain_config.get("event_and_reward", {}).get("failure_flags", {})

        # per-episode bookkeeping for terrain rewards
        # if repeat_reward is False (default), collision/fall rewards are given only once per episode
        self._fallen_reported = False
        self._collision_reported = False

        # build action dims (same as before)
        self.action_dims = {}
        total = 0
        if 'bump' in self.terrain_types:
            self.action_dims['bump'] = 4
            total += 4
        if 'slide_friction' in self.terrain_types:
            self.action_dims['slide_friction'] = 1
            total += 1
        if 'solref' in self.terrain_types:
            self.action_dims['solref'] = 1
            total += 1
        self.total_action_dims = total

        # Terrain changer helper (owns terrain policy and mapping)
        self.terrain_changer = TerrainChanger(self.model, self.data, action_dims=self.action_dims, config_file=terrain_config_file)

        # Observation config for terrain agent.
        obs_cfg = self.terrain_config.get("observation", {})
        self.obs_include_last_action = bool(obs_cfg.get("include_last_action", True))
        self.obs_include_foot_contacts = bool(obs_cfg.get("include_foot_contacts", False))  # TODO 暂定False，相关功能待验证，也可能不需要
        self.obs_contact_force_threshold = float(obs_cfg.get("contact_force_threshold", 1.0))
        local_cfg = obs_cfg.get("local_height_map", {})
        self.local_map_enabled = bool(local_cfg.get("enabled", True))
        self.local_map_size_m = float(local_cfg.get("size_m", 1.0))
        self.local_map_resolution_m = float(local_cfg.get("resolution_m", 0.2))
        self.local_map_side = max(1, int(round(self.local_map_size_m / max(self.local_map_resolution_m, 1e-6))))

        # Cache body ids for simple foot contact state.
        self.foot_body_ids = []
        for body_name in ["FL_calf", "FR_calf", "RL_calf", "RR_calf"]:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if bid >= 0:
                self.foot_body_ids.append(int(bid))

        # Use controller's control decimation and PD gains
        self.control_decimation = self.go2_controller.control_decimation

        self.render = self.terrain_config["visualization"]["render"]
        self.lock_camera = self.terrain_config["visualization"]["lock_camera"]
        self.realtime_sim = self.terrain_config["visualization"]["realtime_sim"]
        self.trace_enabled = bool(self.terrain_config.get("logging", {}).get("enable_trace", False))
        if self.render:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.cam.azimuth = 0
            self.viewer.cam.elevation = -20
            self.viewer.cam.distance = 1.5
            self.viewer.cam.lookat[:] = self.data.qpos[:3]

        self.step_counter = 0
        self.robot_counter = 0

        self.init_skip_time = self.go2_config["init_skip_time"]
        self.init_skip_frame = 10

    def reset(self):
        """Reset physics and counters. Returns initial observation (robot-centric).

        Accept seed/options for now but accept them to be compatible with Gym API
        """
        # ignore seed/options for now but accept them to be compatible with Gym API
        mujoco.mj_resetData(self.model, self.data)
        # recreate terrain_changer with fresh data reference

        self.go2_controller.reset()
        self.terrain_changer.reset(self.data)
        if self.render:
            self.viewer.update_hfield(self.terrain_changer.hfield_id)
            self.viewer.sync()

        self.step_counter = 0
        self.robot_counter = 0
        # reset per-episode flags
        self._fallen_reported = False
        self._collision_reported = False
        # return initial robot observation (single value)

        # 前10帧不控制，且提供2s的保护时间
        target_dof_pos = self.go2_controller.default_angles.copy()
        total_sim_steps = int(self.init_skip_time / self.model.opt.timestep) + self.init_skip_frame  # 2s保护时间 + 10帧控制空窗期
        for sim_i in range(total_sim_steps):
            self.robot_counter += 1
            step_start = time.time()
            # at control boundaries compute new target and tau
            if sim_i >= 10 and sim_i % int(self.control_decimation) == 0:
                target_dof_pos = safe_call(self.go2_controller.compute_action, d=self.data, counter=self.robot_counter)
                # target_dof_pos = self.go2_controller.compute_action(self.data)
            tau = pd_control(target_dof_pos, self.data.qpos[7:], self.go2_controller.kps,
                             np.zeros_like(self.go2_controller.kds), self.data.qvel[6:], self.go2_controller.kds)
            self.data.ctrl[:] = tau
            mujoco.mj_step(self.model, self.data)

            if self.render:
                if self.lock_camera:
                    self.viewer.cam.lookat[:] = self.data.qpos[:3]
                self.viewer.sync()

            if self.realtime_sim:  # 只能确保仿真不比真实世界快，但是可能会比真实世界慢，取决于计算开销
                time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

        return self.get_terrain_observation()

    def close_viewer(self):
        if self.render:
            self.viewer.close()

    def start_viewer(self):
        if self.render:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.cam.azimuth = 0
            self.viewer.cam.elevation = -20
            self.viewer.cam.distance = 1.5
            self.viewer.cam.lookat[:] = self.data.qpos[:3]

    def render_hfield(self):
        if self.render:
            self.viewer.update_hfield(self.terrain_changer.hfield_id)
            self.viewer.sync()

    def step(self, terrain_action):

        self.step_counter += 1

        robot_states = None
        robot_actions = None
        if self.trace_enabled:
            # Full robot rollout trace in this terrain step:
            # states: [s0, s1, ..., sN] ; actions: [a0, ..., a(N-1)]
            robot_obs = safe_call(self.go2_controller.get_observation, d=self.data, counter=self.robot_counter)
            robot_states = [
                {
                    'sim_i': -1,
                    'qpos': np.asarray(self.data.qpos, dtype=np.float64).tolist(),
                    'qvel': np.asarray(self.data.qvel, dtype=np.float64).tolist(),
                    # 'robot_obs': np.asarray(self.go2_controller.get_observation(self.data), dtype=np.float32).tolist(),
                    'robot_obs': np.asarray(robot_obs, dtype=np.float32).tolist(),
                }
            ]
            robot_actions = []

        # apply terrain action via TerrainChanger and remember it
        self.terrain_changer.apply_action_vector(terrain_action)
        self.terrain_changer._refresh_terrain_safe()

        if self.render:
            self.viewer.update_hfield(self.terrain_changer.hfield_id)
            self.viewer.sync()

        total_sim_steps = int(self.terrain_decimation * self.control_decimation)

        # run robot controller with zero-order hold on tau
        target_dof_pos = self.go2_controller.default_angles.copy()
        for sim_i in range(total_sim_steps):
            self.robot_counter += 1
            step_start = time.time()
            # at control boundaries compute new target and tau
            if sim_i % int(self.control_decimation) == 0:
                target_dof_pos = safe_call(self.go2_controller.compute_action, d=self.data, counter=self.robot_counter)
                # target_dof_pos = self.go2_controller.compute_action(self.data)
            tau = pd_control(target_dof_pos, self.data.qpos[7:], self.go2_controller.kps, np.zeros_like(self.go2_controller.kds), self.data.qvel[6:], self.go2_controller.kds)

            if self.trace_enabled:
                robot_actions.append(
                    {
                        'sim_i': int(sim_i),
                        'target_dof_pos': np.asarray(target_dof_pos, dtype=np.float32).tolist(),
                        'tau': np.asarray(tau, dtype=np.float32).tolist(),
                        'policy_action_prev': np.asarray(self.go2_controller.action_policy_prev, dtype=np.float32).tolist(),
                    }
                )

            self.data.ctrl[:] = tau
            mujoco.mj_step(self.model, self.data)

            if self.trace_enabled:
                robot_obs = safe_call(self.go2_controller.get_observation, d=self.data, counter=self.robot_counter)
                robot_states.append(
                    {
                        'sim_i': int(sim_i),
                        'qpos': np.asarray(self.data.qpos, dtype=np.float64).tolist(),
                        'qvel': np.asarray(self.data.qvel, dtype=np.float64).tolist(),
                        # 'robot_obs': np.asarray(self.go2_controller.get_observation(self.data), dtype=np.float32).tolist(),
                        'robot_obs': np.asarray(robot_obs, dtype=np.float32).tolist(),
                    }
                )

            if self.render:
                if self.lock_camera:
                    self.viewer.cam.lookat[:] = self.data.qpos[:3]
                self.viewer.sync()

            if self.realtime_sim:  # 只能确保仿真不比真实世界快，但是可能会比真实世界慢，取决于计算开销
                time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

        # compute terrain reward and next obs
        terrain_reward, terrain_info, done = self.compute_terrain_reward()

        # Update last action before reading next observation so obs can include current terrain action.
        self.terrain_changer.last_action = np.asarray(terrain_action, dtype=np.float32)
        next_terrain_obs = self.get_terrain_observation()

        info = {
            'terrain_reward': float(terrain_reward),
            **terrain_info,
        }
        if self.trace_enabled:
            info['go2_rollout_trace'] = {
                'states': robot_states,
                'actions': robot_actions,
            }
            info['terrain_action'] = np.asarray(terrain_action, dtype=np.float32).tolist()

        # print(f"step_counter: {self.step_counter}, robot_counter: {self.robot_counter}, terrain_reward: {terrain_reward}")
        # if terrain_reward != 0.0:
        if terrain_reward >= 1.0:
            print(f"step_counter: {self.step_counter}, robot_counter: {self.robot_counter}, terrain_reward: {terrain_reward}, info: {terrain_info}")

        return next_terrain_obs, np.asarray(terrain_action, dtype=np.float32), float(terrain_reward), done, info

    def step_only_robot(self, terrain_decimation=1):
        target_dof_pos = self.go2_controller.default_angles.copy()
        for sim_i in range(self.control_decimation * terrain_decimation):
            self.robot_counter += 1
            step_start = time.time()
            # at control boundaries compute new target and tau
            if sim_i % int(self.control_decimation) == 0:
                target_dof_pos = safe_call(self.go2_controller.compute_action, d=self.data, counter=self.robot_counter)
                # target_dof_pos = self.go2_controller.compute_action(self.data)
            tau = pd_control(target_dof_pos, self.data.qpos[7:], self.go2_controller.kps, np.zeros_like(self.go2_controller.kds), self.data.qvel[6:], self.go2_controller.kds)

            self.data.ctrl[:] = tau
            mujoco.mj_step(self.model, self.data)

            if self.render:
                if self.lock_camera:
                    self.viewer.cam.lookat[:] = self.data.qpos[:3]
                self.viewer.sync()

            if self.realtime_sim:  # 只能确保仿真不比真实世界快，但是可能会比真实世界慢，取决于计算开销
                time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

        # compute terrain reward and next obs
        terrain_reward, terrain_info, done = self.compute_terrain_reward()

        # Update last action before reading next observation so obs can include current terrain action.
        # self.terrain_changer.last_action = np.asarray(terrain_action, dtype=np.float32)
        next_terrain_obs = self.get_terrain_observation()

        info = {
            'terrain_reward': float(terrain_reward),
            **terrain_info,
        }

        # print(f"step_counter: {self.step_counter}, robot_counter: {self.robot_counter}, terrain_reward: {terrain_reward}")
        # if terrain_reward != 0.0:
        if terrain_reward >= 1.0:
            print(f"step_counter: {self.step_counter}, robot_counter: {self.robot_counter}, terrain_reward: {terrain_reward}, info: {terrain_info}")

        return next_terrain_obs, None, float(terrain_reward), done, info

    # TODO 梅花桩相关
    def set_robot_spawn_pose(self, x=0.0, y=0.0, z=None, yaw=0.0):
        """Set robot root pose before rollouts (for pile tests and scripted starts)."""
        self.data.qpos[0] = float(x)
        self.data.qpos[1] = float(y)
        if z is not None:
            self.data.qpos[2] = float(z)

        # roll=pitch=0 quaternion from yaw.
        cy = np.cos(0.5 * float(yaw))
        sy = np.sin(0.5 * float(yaw))
        self.data.qpos[3:7] = np.array([cy, 0.0, 0.0, sy], dtype=np.float64)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

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
        if not self.local_map_enabled:
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

    def get_terrain_observation(self):
        """Terrain obs = robot obs + optional foot contacts + optional local map + optional last terrain action."""
        robot_obs = safe_call(self.go2_controller.get_observation_without_prev_action, d=self.data, counter=self.robot_counter)
        # robot_obs = self.go2_controller.get_observation(self.data).astype(np.float32)
        # robot_obs = robot_obs[:len(robot_obs) - self.go2_controller.num_actions]  # TODO 去掉robot obs中的last action部分
        chunks = [robot_obs]

        if self.obs_include_foot_contacts:
            chunks.append(self._get_foot_contact_flags())

        local_map_obs = self._get_local_height_map_obs()
        if local_map_obs.size > 0:
            chunks.append(local_map_obs.astype(np.float32))

        if self.obs_include_last_action and self.total_action_dims > 0:
            chunks.append(self.terrain_changer.last_action.astype(np.float32))

        return np.concatenate(chunks, axis=0)

    # ---------- terrain reward helpers ----------
    def _collect_motion_state(self) -> Tuple[float, float, float, float, float]:
        base_z = float(self.data.qpos[2])
        lin_vel = float(np.linalg.norm(self.data.qvel[:2]))
        quat = self.data.qpos[3:7]
        roll, pitch, _ = quat_to_rpy(quat)
        target_speed = float(self.terrain_config["event_and_reward"]["target_speed"])
        return base_z, lin_vel, float(roll), float(pitch), target_speed

    def _is_failure_enabled(self, key: str) -> bool:
        cfg = self.failure_flags if isinstance(self.failure_flags, dict) else {}
        if key in cfg:
            return bool(cfg.get(key, True))
        return True

    def _get_ground_height_at_xy(self, x: float, y: float) -> float:
        """Estimate terrain surface height at world XY from the active hfield."""
        hfield_id = int(self.terrain_changer.hfield_id)
        nrow = int(self.model.hfield_nrow[hfield_id])
        ncol = int(self.model.hfield_ncol[hfield_id])

        size_x = float(self.model.hfield_size[hfield_id][0])  # half-size x
        size_y = float(self.model.hfield_size[hfield_id][1])  # half-size y
        z_scale = float(self.model.hfield_size[hfield_id][2])  # TODO 所有涉及到高度的地方都应该乘以z_scale，目前地图设置z_scale=1所以不用乘

        center_x = float(self.model.geom_pos[self.terrain_changer.geom_id][0])
        center_y = float(self.model.geom_pos[self.terrain_changer.geom_id][1])
        center_z = float(self.model.geom_pos[self.terrain_changer.geom_id][2])

        # map world xy -> normalized [0,1] inside hfield footprint
        u = (x - center_x + size_x) / (2.0 * size_x)
        v = (y - center_y + size_y) / (2.0 * size_y)
        u = float(np.clip(u, 0.0, 1.0))
        v = float(np.clip(v, 0.0, 1.0))

        col = int(np.clip(round(u * (ncol - 1)), 0, ncol - 1))
        row = int(np.clip(round(v * (nrow - 1)), 0, nrow - 1))

        # model.hfield_data stores height samples; world z is scaled by size[2]
        h = float(self.terrain_changer.hfield[row, col])
        return center_z + z_scale * h

    def _is_fallen(self, base_z: float, roll: float, pitch: float) -> bool:
        if not self._is_failure_enabled("fallen"):
            return False

        x = float(self.data.qpos[0])
        y = float(self.data.qpos[1])
        ground_z = self._get_ground_height_at_xy(x, y)
        base_rel_height = base_z - ground_z

        if base_rel_height < float(self.terrain_config["event_and_reward"]["fall_height_threshold"]):
            return True
        angle_thresh = float(self.terrain_config["event_and_reward"]["fall_angle_threshold"])
        return abs(roll) > angle_thresh or abs(pitch) > angle_thresh

    def _compute_fall_reward(self, fallen: bool, repeat: bool) -> float:
        if not fallen:
            return 0.0
        if repeat or not self._fallen_reported:
            self._fallen_reported = True
            return float(self.terrain_config["event_and_reward"]["fall_reward"])
        return 0.0

    def _analyze_contacts(self) -> Tuple[bool, bool, bool]:
        if not (
            self._is_failure_enabled("collided")
            or self._is_failure_enabled("base_collision")
            or self._is_failure_enabled("thigh_collision")
        ):
            return False, False, False

        collided = False
        base_collision = False
        thigh_collision = False
        force_thresh = float(self.terrain_config["event_and_reward"]["collision_force_threshold"])

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, i, force)
            f_mag = np.linalg.norm(force[:3])
            if f_mag < force_thresh:
                continue

            collided = True
            geom1 = contact.geom1
            geom2 = contact.geom2
            body1 = self.model.geom_bodyid[geom1]
            body2 = self.model.geom_bodyid[geom2]
            name1 = self.model.body(body1).name
            name2 = self.model.body(body2).name

            if "base" in name1 or "base" in name2:
                base_collision = True
            if "thigh" in name1 or "thigh" in name2:
                thigh_collision = True

        collided = collided and self._is_failure_enabled("collided")
        base_collision = base_collision and self._is_failure_enabled("base_collision")
        thigh_collision = thigh_collision and self._is_failure_enabled("thigh_collision")

        return collided, base_collision, thigh_collision

    def _compute_collision_reward(self, collided: bool, base_collision: bool, thigh_collision: bool, repeat: bool) -> float:
        reward = 0.0
        if base_collision and (repeat or not self._collision_reported):
            reward += float(self.terrain_config["event_and_reward"]["base_collision_reward"])
            self._collision_reported = True
        if thigh_collision:
            reward += float(self.terrain_config["event_and_reward"]["thigh_collision_reward"])
        if collided:
            reward += float(self.terrain_config["event_and_reward"]["collision_reward"])
        return reward

    def _compute_tilt_reward(self, roll: float, pitch: float) -> Tuple[float, float]:
        tilt = abs(roll) + abs(pitch)
        reward = float(self.terrain_config["event_and_reward"]["tilt_reward_scale"]) * tilt
        return float(tilt), float(reward)

    def _compute_speed_reward(self, lin_vel: float, target_speed: float) -> float:
        speed_loss = max(0.0, target_speed - lin_vel)
        return float(self.terrain_config["event_and_reward"]["speed_reward_scale"]) * speed_loss

    def _compute_stuck_reward(self, lin_vel: float, target_speed: float) -> Tuple[bool, float]:
        if not self._is_failure_enabled("stuck"):
            return False, 0.0
        stuck = lin_vel < float(self.terrain_config["event_and_reward"]["stuck_speed_threshold"]) and target_speed > 0.2
        if not stuck:
            return False, 0.0
        return True, float(self.terrain_config["event_and_reward"]["stuck_reward"])

    def _is_out_of_terrain_edge(self) -> bool:
        """Check whether robot base xy leaves terrain bounds (with optional margin)."""
        x = float(self.data.qpos[0])
        y = float(self.data.qpos[1])

        size_x = self.terrain_changer.terrain_size_x
        size_y = self.terrain_changer.terrain_size_y
        center_x = self.terrain_changer.terrain_center_x
        center_y = self.terrain_changer.terrain_center_y

        margin = self.terrain_config["termination"]["terrain_edge_margin"]

        half_x = size_x * 0.5 - margin
        half_y = size_y * 0.5 - margin
        if half_x <= 0.0 or half_y <= 0.0:
            return False

        return (abs(x - center_x) >= half_x) or (abs(y - center_y) >= half_y)

    # TODO 增加stuck TODO 111
    def _compute_done(self, fallen: bool, base_collision: bool, out_of_terrain_edge: bool) -> bool:
        if self._is_failure_enabled("fallen") and fallen and self.terrain_config["termination"]["terminate_on_fall"]:
            return True
        if self._is_failure_enabled("base_collision") and base_collision and self.terrain_config["termination"]["terminate_on_base_collision"]:
            return True
        if self._is_failure_enabled("stuck") and base_collision and self.terrain_config["termination"]["terminate_on_stuck"]:
            return True
        if out_of_terrain_edge and self.terrain_config["termination"]["terminate_on_terrain_edge"]:
            return True

        return False

    def compute_terrain_reward(self) -> Tuple[float, dict, bool]:
        reward = 0.0
        repeat = bool(self.terrain_config["event_and_reward"]["repeat_reward"])

        base_z, lin_vel, roll, pitch, target_speed = self._collect_motion_state()

        # 1. 判断是否跌倒并计算跌倒奖励
        fallen = self._is_fallen(base_z, roll, pitch)
        reward += self._compute_fall_reward(fallen, repeat)

        # 2. 分析接触信息并计算碰撞奖励
        collided, base_collision, thigh_collision = self._analyze_contacts()
        reward += self._compute_collision_reward(collided, base_collision, thigh_collision, repeat)

        # 3. 计算倾斜奖励
        tilt, tilt_reward = self._compute_tilt_reward(roll, pitch)
        reward += tilt_reward

        # 4. 计算速度奖励
        reward += self._compute_speed_reward(lin_vel, target_speed)

        # 5. 计算被卡住奖励
        stuck, stuck_reward = self._compute_stuck_reward(lin_vel, target_speed)
        reward += stuck_reward

        # 6. 继续判断是否到达地形边缘，并计算done条件
        out_of_terrain_edge = self._is_out_of_terrain_edge()
        done = self._compute_done(fallen, base_collision, out_of_terrain_edge)

        x = float(self.data.qpos[0])
        y = float(self.data.qpos[1])
        ground_height = self._get_ground_height_at_xy(x, y)
        base_rel_height = base_z - ground_height

        info = {
            "fallen": fallen,
            "collided": collided,
            "base_collision": base_collision,
            "thigh_collision": thigh_collision,
            "stuck": stuck,
            "tilt": tilt,
            "speed": lin_vel,
            "base_height": base_z,
            "ground_height": ground_height,
            "base_rel_height": base_rel_height,
            "out_of_terrain_edge": out_of_terrain_edge,
            "terrain_obs_dim": int(self.get_terrain_observation().shape[0]),
        }

        return reward, info, done


class TerrainGymEnv(gym.Env):
    """Gym wrapper around TerrainTrainer for on-policy RL (terrain agent).

    Observation: terrain observation (robot state + last terrain action)
    Action: flat terrain action vector in [-1,1]^action_dim
    Reward: terrain reward (collision/fall)
    """
    def __init__(self, trainer: 'TerrainTrainer', max_episode_steps: int = 1000):
        super().__init__()
        self.trainer = trainer
        obs_dim = trainer.get_terrain_observation().shape[0]
        act_dim = trainer.total_action_dims
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        if act_dim > 0:
            self.action_space = Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32)
        else:
            self.action_space = Box(low=-1.0, high=1.0, shape=(0,), dtype=np.float32)
        self.max_episode_steps = max_episode_steps
        self._step_count = 0

    def reset(self, *, seed=None, options=None):
        # call trainer.reset in a way that is compatible with multiple trainer implementations
        res = self.trainer.reset()

        # Normalize return to (obs, info)
        if isinstance(res, tuple):
            if len(res) == 2:
                obs, info = res
            elif len(res) == 1:
                obs = res[0]
                info = {}
            else:
                # unexpected extra return values: take first two
                obs = res[0]
                info = res[1] if isinstance(res[1], dict) else {}
        else:
            obs = res
            info = {}

        self._step_count = 0
        return obs, info

    def step(self, action):
        next_obs, act, reward, done, info = self.trainer.step(terrain_action=action)
        self._step_count += 1

        # Determine truncation (time limit) vs termination (env terminal)
        truncated = False
        if self._step_count >= self.max_episode_steps:
            truncated = True

        terminated = bool(done)

        # Always return the 5-tuple (obs, reward, terminated, truncated, info)
        return next_obs, float(reward), bool(terminated), bool(truncated), info

    def render(self, mode='human'):
        # trainer handles rendering via its `render` flag and external viewer
        pass


if __name__ == "__main__":
    ...
