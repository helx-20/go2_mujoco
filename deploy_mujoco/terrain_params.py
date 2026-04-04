import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

import mujoco
import mujoco.viewer
import numpy as np
import yaml


class TerrainChanger:
    def __init__(self, model, data, action_dims=None, config_file=None):
        self.model = model
        self.data = data
        # load default config from yaml on disk
        with open(f"{os.path.dirname(os.path.realpath(__file__))}/{config_file}", "r", encoding="utf-8") as f:
            self.terrain_config = yaml.load(f, Loader=yaml.FullLoader)

        self.hfield_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, "terrain_hfield")
        self.nrow, self.ncol = model.hfield_nrow[self.hfield_id], model.hfield_ncol[self.hfield_id]
        self.hfield = model.hfield_data.reshape(self.nrow, self.ncol)
        self.geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "terrain")

        self.original_hfield = model.hfield_data.copy()

        # action space description (dict): e.g. {'bump':4, 'slide_friction':1}
        self.action_dims = action_dims or {}
        self.total_action_dims = sum(self.action_dims.values()) if self.action_dims else 0

        # These are approximate world sizes for mapping between grid and world coordinates.
        # Prefer scene-provided sizes (geom/hfield size half-extents), otherwise fall back to defaults.
        # try to read geom size (half-extents in MuJoCo)
        half_x = float(self.model.geom_size[self.geom_id][0])
        half_y = float(self.model.geom_size[self.geom_id][1])
        self.terrain_size_x = 2.0 * half_x
        self.terrain_size_y = 2.0 * half_y

        # try to read geom position as center
        self.terrain_center_x = float(self.model.geom_pos[self.geom_id][0])
        self.terrain_center_y = float(self.model.geom_pos[self.geom_id][1])

        self.last_action = np.zeros((self.total_action_dims,), dtype=np.float32)

        self.grid_resolution = self.terrain_size_x / self.ncol

        # Cache for plum-blossom pile indices.
        self._pile_regions = {}
        self._pile_shape = (0, 0)

    def reset(self, mujoco_data):
        self.data = mujoco_data
        self.model.hfield_data[:] = self.original_hfield
        mujoco.mj_setConst(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_step(self.model, self.data)
        self.last_action = np.zeros((self.total_action_dims,), dtype=np.float32)

    def run(self):
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            step = 0

            # self.generate_trig_terrain([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 0, 1]])
            # viewer.update_hfield(self.hfield_id)
            # mujoco.mj_setConst(self.model, self.data)
            # mujoco.mj_forward(self.model, self.data)
            # mujoco.mj_step(self.model, self.data)

            while viewer.is_running():
                if step % 20000 == 0:

                    # 测试用
                    # action = ...
                    # action = np.random.uniform(-1.0, 1.0, (self.total_action_dims,))
                    # self.apply_action_vector(action)

                    viewer.update_hfield(self.hfield_id)

                    mujoco.mj_setConst(self.model, self.data)
                    mujoco.mj_forward(self.model, self.data)
                    mujoco.mj_step(self.model, self.data)
                    viewer.sync()

                step += 1

    def _world_to_grid(self, x, y):
        grid_x = int((x - self.terrain_center_x + self.terrain_size_x / 2) / self.terrain_size_x * self.ncol)
        grid_y = int((y - self.terrain_center_y + self.terrain_size_y / 2) / self.terrain_size_y * self.nrow)
        return np.clip(grid_x, 0, self.nrow - 1), np.clip(grid_y, 0, self.ncol - 1)

    def _norm01_to_world(self, x01, y01):
        x01 = float(np.clip(x01, 0.0, 1.0))
        y01 = float(np.clip(y01, 0.0, 1.0))
        x = self.terrain_center_x - self.terrain_size_x * 0.5 + x01 * self.terrain_size_x
        y = self.terrain_center_y - self.terrain_size_y * 0.5 + y01 * self.terrain_size_y
        return float(x), float(y)

    def _grid_resolution_xy(self):
        grid_res_x = self.terrain_size_x / float(self.ncol)
        grid_res_y = self.terrain_size_y / float(self.nrow)
        return float(grid_res_x), float(grid_res_y)

    def _refresh_terrain_safe(self):
        # --- [步骤 A] 备份当前机器人状态 ---
        qpos_backup = self.data.qpos.copy()
        qvel_backup = self.data.qvel.copy()
        act_backup = self.data.act.copy()

        mujoco.mj_setConst(self.model, self.data)

        # --- 还原状态 ---
        self.data.qpos[:] = qpos_backup
        self.data.qvel[:] = qvel_backup
        self.data.act[:] = act_backup

        # mujoco.mj_forward(self.model, self.data)  # 修改地形后不直接做step，step在主逻辑中操作

    def _refresh_terrain(self):
        mujoco.mj_setConst(self.model, self.data)

    def _lift_robot_if_needed(self, foot_clearance_m):
        # Prevent severe penetration when terrain is suddenly raised under the robot.
        x = float(self.data.qpos[0])
        y = float(self.data.qpos[1])
        gx, gy = self._world_to_grid(x, y)
        local_h = float(self.hfield[gx, gy])
        terrain_z = float(self.model.geom_pos[self.geom_id][2]) + float(self.model.hfield_size[self.hfield_id][2]) * local_h
        min_base_z = terrain_z + foot_clearance_m
        if float(self.data.qpos[2]) < min_base_z:
            self.data.qpos[2] = min_base_z
            mujoco.mj_forward(self.model, self.data)

    def apply_action_vector(self, action):
        """
        Interpret a flat action vector according to self.action_dims and call
        the appropriate setters (set_bump, set_slide_friction, set_solref).
        """
        action = np.asarray(action, dtype=np.float32).reshape(-1)

        idx = 0

        if 'bump' in self.action_dims:
            cx_norm = float(action[idx + 0])
            cy_norm = float(action[idx + 1])
            radius = float(action[idx + 2])
            height = float(action[idx + 3])

            # map normalized cx/cy to world coordinates
            # longitudinal distance ahead
            # prefer robot base pos and velocity if available; otherwise use scene origin
            robot_xy = self.data.qpos[:2].copy()
            lin_vel = self.data.qvel[:2].copy()
            speed = np.linalg.norm(lin_vel)

            # 确定速度方向 dir_f
            if speed > 1e-3:
                dir_f = lin_vel / speed
            else:
                qw, qx, qy, qz = self.data.qpos[3:7]
                yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
                dir_f = np.array([np.cos(yaw), np.sin(yaw)])

            # dump 相对坐标圆心[dist, lat]
            min_forward_dist = self.terrain_config['terrain_action']['min_forward_dist']
            max_forward_dist = self.terrain_config['terrain_action']['max_forward_dist']
            max_lateral = self.terrain_config['terrain_action']['max_lateral']
            max_bump_height = self.terrain_config['terrain_action']['max_bump_height']

            dist = min_forward_dist + (cx_norm + 1.0) / 2.0 * (max_forward_dist - min_forward_dist)
            lat = cy_norm * max_lateral
            perp = np.array([-dir_f[1], dir_f[0]])
            target_xy = robot_xy + dir_f * dist + perp * lat
            gx, gy = self._world_to_grid(float(target_xy[0]), float(target_xy[1]))

            radius_min = self.terrain_config['terrain_action']['radius_min']
            radius_max = self.terrain_config['terrain_action']['radius_max']
            radius_grid_min = radius_min / self.grid_resolution
            radius_grid_max = radius_max / self.grid_resolution
            # scale radius to configured grid units
            radius_scaled = (radius + 1.0) / 2.0 * (radius_grid_max - radius_grid_min) + radius_grid_min
            # scale height to configured max
            height_scaled = float(height) * max_bump_height

            # 保护机器人所在区域不被修改
            robot_gx, robot_gy = self._world_to_grid(float(robot_xy[0]), float(robot_xy[1]))
            self.set_bump(
                int(gx),
                int(gy),
                float(radius_scaled),
                float(height_scaled),
                int(robot_gx),
                int(robot_gy)
            )

            idx += self.action_dims['bump']

        if 'slide_friction' in self.action_dims:
            mu = float(action[idx])
            mu_scaled = (mu + 1.0) / 2.0 * 1.5 + 0.1
            self.set_slide_friction(float(mu_scaled))
            idx += self.action_dims['slide_friction']

        if 'solref' in self.action_dims:
            sol = float(action[idx])
            solref_scaled = (sol + 1.0) / 2.0 * 0.5 + 0.01
            self.set_solref(float(solref_scaled))
            idx += self.action_dims['solref']

        # self.last_action = np.asarray(action, dtype=np.float32).reshape(-1)

    def apply_action_vector_with_restore(self, action):
        """
        Interpret a flat action vector according to self.action_dims and call
        the appropriate setters (set_bump, set_slide_friction, set_solref).
        """
        action = np.asarray(action, dtype=np.float32).reshape(-1)

        idx = 0

        if 'bump' in self.action_dims:
            cx_norm = float(action[idx + 0])
            cy_norm = float(action[idx + 1])
            radius = float(action[idx + 2])
            height = float(action[idx + 3])

            # map normalized cx/cy to world coordinates
            # longitudinal distance ahead
            # prefer robot base pos and velocity if available; otherwise use scene origin
            robot_xy = self.data.qpos[:2].copy()
            lin_vel = self.data.qvel[:2].copy()
            speed = np.linalg.norm(lin_vel)

            # 确定速度方向 dir_f
            if speed > 1e-3:
                dir_f = lin_vel / speed
            else:
                qw, qx, qy, qz = self.data.qpos[3:7]
                yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
                dir_f = np.array([np.cos(yaw), np.sin(yaw)])

            # dump 相对坐标圆心[dist, lat]
            min_forward_dist = self.terrain_config['terrain_action']['min_forward_dist']
            max_forward_dist = self.terrain_config['terrain_action']['max_forward_dist']
            max_lateral = self.terrain_config['terrain_action']['max_lateral']
            max_bump_height = self.terrain_config['terrain_action']['max_bump_height']

            dist = min_forward_dist + (cx_norm + 1.0) / 2.0 * (max_forward_dist - min_forward_dist)
            lat = cy_norm * max_lateral
            perp = np.array([-dir_f[1], dir_f[0]])
            target_xy = robot_xy + dir_f * dist + perp * lat
            gx, gy = self._world_to_grid(float(target_xy[0]), float(target_xy[1]))

            radius_min = self.terrain_config['terrain_action']['radius_min']
            radius_max = self.terrain_config['terrain_action']['radius_max']
            radius_grid_min = radius_min / self.grid_resolution
            radius_grid_max = radius_max / self.grid_resolution
            # scale radius to configured grid units
            radius_scaled = (radius + 1.0) / 2.0 * (radius_grid_max - radius_grid_min) + radius_grid_min
            # scale height to configured max
            height_scaled = float(height) * max_bump_height

            # 保护机器人所在区域不被修改
            robot_gx, robot_gy = self._world_to_grid(float(robot_xy[0]), float(robot_xy[1]))
            restore_info = self.set_bump_with_restore(
                int(gx),
                int(gy),
                float(radius_scaled),
                float(height_scaled),
                int(robot_gx),
                int(robot_gy)
            )

            idx += self.action_dims['bump']

            return restore_info

        if 'slide_friction' in self.action_dims:
            return None

        if 'solref' in self.action_dims:
            return None

        return None

    def apply_action_vector_with_robot(self, qpos, qvel, action):
        """
        Interpret a flat action vector according to self.action_dims and call
        the appropriate setters (set_bump, set_slide_friction, set_solref).
        """
        action = np.asarray(action, dtype=np.float32).reshape(-1)

        idx = 0

        if 'bump' in self.action_dims:
            cx_norm = float(action[idx + 0])
            cy_norm = float(action[idx + 1])
            radius = float(action[idx + 2])
            height = float(action[idx + 3])

            # map normalized cx/cy to world coordinates
            # longitudinal distance ahead
            # prefer robot base pos and velocity if available; otherwise use scene origin
            robot_xy = qpos[:2].copy()
            lin_vel = qvel[:2].copy()
            speed = np.linalg.norm(lin_vel)

            # 确定速度方向 dir_f
            if speed > 1e-3:
                dir_f = lin_vel / speed
            else:
                qw, qx, qy, qz = qpos[3:7]
                yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
                dir_f = np.array([np.cos(yaw), np.sin(yaw)])

            # dump 相对坐标圆心[dist, lat]
            min_forward_dist = self.terrain_config['terrain_action']['min_forward_dist']
            max_forward_dist = self.terrain_config['terrain_action']['max_forward_dist']
            max_lateral = self.terrain_config['terrain_action']['max_lateral']
            max_bump_height = self.terrain_config['terrain_action']['max_bump_height']

            dist = min_forward_dist + (cx_norm + 1.0) / 2.0 * (max_forward_dist - min_forward_dist)
            lat = cy_norm * max_lateral
            perp = np.array([-dir_f[1], dir_f[0]])
            target_xy = robot_xy + dir_f * dist + perp * lat
            gx, gy = self._world_to_grid(float(target_xy[0]), float(target_xy[1]))

            radius_min = self.terrain_config['terrain_action']['radius_min']
            radius_max = self.terrain_config['terrain_action']['radius_max']
            radius_grid_min = radius_min / self.grid_resolution
            radius_grid_max = radius_max / self.grid_resolution
            # scale radius to configured grid units
            radius_scaled = (radius + 1.0) / 2.0 * (radius_grid_max - radius_grid_min) + radius_grid_min
            # scale height to configured max
            height_scaled = float(height) * max_bump_height

            # 保护机器人所在区域不被修改
            robot_gx, robot_gy = self._world_to_grid(float(robot_xy[0]), float(robot_xy[1]))
            self.set_bump(
                int(gx),
                int(gy),
                float(radius_scaled),
                float(height_scaled),
                int(robot_gx),
                int(robot_gy)
            )

            idx += self.action_dims['bump']

        if 'slide_friction' in self.action_dims:
            mu = float(action[idx])
            mu_scaled = (mu + 1.0) / 2.0 * 1.5 + 0.1
            self.set_slide_friction(float(mu_scaled))
            idx += self.action_dims['slide_friction']

        if 'solref' in self.action_dims:
            sol = float(action[idx])
            solref_scaled = (sol + 1.0) / 2.0 * 0.5 + 0.01
            self.set_solref(float(solref_scaled))
            idx += self.action_dims['solref']

        # self.last_action = np.asarray(action, dtype=np.float32).reshape(-1)

    def set_bump(self, gx, gy, radius, height, robot_gx, robot_gy):

        # ['terrain_action']['no_change_radius']
        no_change_radius = self.terrain_config.get('terrain_action', {}).get('no_change_radius', False)
        no_change_radius_grid = no_change_radius / self.grid_resolution

        for row in range(self.nrow):
            for col in range(self.ncol):

                dx = col - gx
                dy = row - gy
                dist = np.sqrt(dx * dx + dy * dy)

                if dist < radius:

                    # ===== 机器人保护区域 =====
                    dx_r = col - robot_gx
                    dy_r = row - robot_gy
                    dist_robot = np.sqrt(dx_r * dx_r + dy_r * dy_r)

                    if no_change_radius and dist_robot < no_change_radius_grid:
                        continue  # 不允许修改

                    self.hfield[row, col] = height * np.exp(
                        -dist ** 2 / (2 * radius ** 2)
                    )

    def set_slide_friction(self, mu):
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "terrain")
        self.model.geom_friction[geom_id][0] = mu

    def set_solref(self, solref):
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "terrain")
        self.model.geom_solref[geom_id] = np.array([solref, 1.0])

    def set_bump_with_restore(self, gx, gy, radius, height, robot_gx, robot_gy):
        restore_info = []

        # ['terrain_action']['no_change_radius']
        no_change_radius = self.terrain_config.get('terrain_action', {}).get('no_change_radius', False)
        no_change_radius_grid = no_change_radius / self.grid_resolution

        for row in range(self.nrow):
            for col in range(self.ncol):

                dx = col - gx
                dy = row - gy
                dist = np.sqrt(dx * dx + dy * dy)

                if dist < radius:

                    # ===== 机器人保护区域 =====
                    dx_r = col - robot_gx
                    dy_r = row - robot_gy
                    dist_robot = np.sqrt(dx_r * dx_r + dy_r * dy_r)

                    if no_change_radius and dist_robot < no_change_radius_grid:
                        continue  # 不允许修改

                    restore_info.append((row, col, self.hfield[row, col]))  # 记录修改前的高度值，以便后续恢复
                    self.hfield[row, col] = height * np.exp(
                        -dist ** 2 / (2 * radius ** 2)
                    )
        return restore_info

    def set_restore_bump(self, restore_info):
        for row, col, original_height in restore_info:
            self.hfield[row, col] = original_height

    def enforce_safe_spawn_area(self, center_world=(0.0, 0.0), safe_radius_m=1.0, blend_radius_m=1.0, target_height=0.0):
        """
        Make a safe spawn area around center_world with smooth transition:
        - r <= safe_radius_m: fully flattened to target_height
        - safe_radius_m < r < safe_radius_m + blend_radius_m: smooth blend
        - outside: keep original terrain
        """
        cx, cy = center_world
        gx, gy = self._world_to_grid(cx, cy)

        # meter -> grid (use x/y average to keep isotropic disk)
        grid_res_x = self.terrain_size_x / self.ncol
        grid_res_y = self.terrain_size_y / self.nrow
        grid_res = 0.5 * (grid_res_x + grid_res_y)

        safe_r = safe_radius_m / grid_res
        blend_r = max(blend_radius_m / grid_res, 1e-6)

        rows = np.arange(self.nrow, dtype=np.float32)[:, None]
        cols = np.arange(self.ncol, dtype=np.float32)[None, :]
        dist = np.sqrt((rows - gx) ** 2 + (cols - gy) ** 2)

        # t: 0 in safe core, 1 outside blend ring
        t = np.clip((dist - safe_r) / blend_r, 0.0, 1.0)
        # smoothstep for C1-like continuous transition
        w = t * t * (3.0 - 2.0 * t)

        self.hfield[:, :] = (1.0 - w) * target_height + w * self.hfield[:, :]
        self._refresh_terrain_safe()

    # TODO 初始化速度很慢
    def generate_bumps_terrain(self, bumps_array, safe_pos=(0.0, 0.0), safe_radius=0):
        """Initialize terrain from a bumps list.

        bumps_array format per item:
          (x01, y01, h) or (x01, y01, h, radius_m)
        where x01/y01 are normalized in [0,1].
        """
        self.hfield[:, :] = 0.0
        grid_res_x, grid_res_y = self._grid_resolution_xy()
        grid_res = 0.5 * (grid_res_x + grid_res_y)

        for item in bumps_array:
            print(item)
            x01, y01, h_m, radius_m = float(item[0]), float(item[1]), float(item[2]), float(item[3])
            wx, wy = self._norm01_to_world(x01, y01)
            gx, gy = self._world_to_grid(wx, wy)
            radius_grid = max(1.0, radius_m / max(grid_res, 1e-8))

            for row in range(self.nrow):
                for col in range(self.ncol):
                    dx = col - gy
                    dy = row - gx
                    dist = np.sqrt(dx * dx + dy * dy)
                    if dist > radius_grid:
                        continue
                    bump_h = h_m * np.exp(-dist * dist / (2.0 * radius_grid * radius_grid))
                    # Keep larger height if multiple bumps overlap.
                    self.hfield[row, col] = max(self.hfield[row, col], bump_h)

        self.enforce_safe_spawn_area(
            center_world=safe_pos,
            safe_radius_m=float(safe_radius),
            blend_radius_m=max(0.2, float(safe_radius)),
            target_height=0.0,
        )
        self._refresh_terrain_safe()

    def generate_trig_terrain(self, angle_array):
        angle_array = np.array(angle_array)

        # Current scene resolution.
        nx = int(self.ncol)
        ny = int(self.nrow)
        my, mx, _ = angle_array.shape

        # Uniform phase grids over [0, 2pi).
        gx = np.linspace(0.0, 2.0 * np.pi, nx, endpoint=False, dtype=np.float32)
        gy = np.linspace(0.0, 2.0 * np.pi, ny, endpoint=False, dtype=np.float32)
        X, Y = np.meshgrid(gx, gy)  # [ny, nx]

        terrain = np.zeros((ny, nx), dtype=np.float32)

        # Use angle-array indices as harmonic IDs.
        for iy in range(my):
            fy = iy + 1
            for ix in range(mx):
                fx = ix + 1
                theta, alt = angle_array[iy][ix]
                terrain += (
                    np.sin(fx * X + theta)
                    + 0.5 * np.cos(fy * Y - theta)
                    + 0.35 * np.sin(fx * X + fy * Y + theta)
                ) * alt

        terrain /= float(max(1, mx * my))

        # Normalize then scale to target amplitude.
        max_abs = float(np.max(np.abs(terrain)))
        if max_abs > 1e-8:
            terrain = terrain / max_abs

        # Write back to hfield.
        self.hfield[:, :] = terrain
        self._refresh_terrain_safe()

    def generate_plum_blossom_piles(self, num_x, num_y, center_world=(0.0, 0.0), base_height=0.1, pile_size_m=0.08, gap_m=0.01):
        """Generate rectangular plum-blossom piles and cache each pile region.

        num_x/num_y: pile count on x/y directions.
        center_world: world-space center of pile field.
        base_height: pile top height in hfield units.
        """
        num_x = int(max(1, num_x))
        num_y = int(max(1, num_y))
        base_height = float(base_height)

        self.hfield[:, :] = 0.0
        self._pile_regions = {}
        self._pile_shape = (num_x, num_y)

        grid_res_x, grid_res_y = self._grid_resolution_xy()
        pile_w = max(1, int(round(float(pile_size_m) / max(grid_res_x, 1e-8))))
        pile_h = max(1, int(round(float(pile_size_m) / max(grid_res_y, 1e-8))))
        gap_x = max(1, int(round(float(gap_m) / max(grid_res_x, 1e-8))))
        gap_y = max(1, int(round(float(gap_m) / max(grid_res_y, 1e-8))))
        # gap_x = 0
        # gap_y = 0

        cx, cy = self._world_to_grid(float(center_world[0]), float(center_world[1]))
        total_w = num_x * pile_w + (num_x - 1) * gap_x
        total_h = num_y * pile_h + (num_y - 1) * gap_y
        start_row = int(cx - total_h // 2)
        start_col = int(cy - total_w // 2)

        for iy in range(num_y):
            for ix in range(num_x):
                r0 = start_row + iy * (pile_h + gap_y)
                c0 = start_col + ix * (pile_w + gap_x)
                r1 = int(np.clip(r0 + pile_h, 0, self.nrow))
                c1 = int(np.clip(c0 + pile_w, 0, self.ncol))
                r0 = int(np.clip(r0, 0, self.nrow))
                c0 = int(np.clip(c0, 0, self.ncol))
                if r1 <= r0 or c1 <= c0:
                    continue
                self.hfield[r0:r1, c0:c1] = base_height
                self._pile_regions[(ix, iy)] = (r0, r1, c0, c1)

        self._refresh_terrain_safe()

    # TODO 检查是否有问题
    def update_plum_blossom_piles(self, control_list):
        """Control pile heights.

        control_list item format: (ix, iy, delta_h)
        """
        for item in control_list:
            if len(item) != 3:
                continue
            ix, iy, dh = int(item[0]), int(item[1]), float(item[2])
            key = (ix, iy)
            if key not in self._pile_regions:
                continue
            r0, r1, c0, c1 = self._pile_regions[key]
            self.hfield[r0:r1, c0:c1] += dh

        foot_clearance_m = float(self.terrain_config.get("plum_blossom", {}).get("foot_clearance_m", 0))
        self._lift_robot_if_needed(foot_clearance_m)
        self._refresh_terrain_safe()


if __name__ == "__main__":

    # yaml 读取测试
    config_file = "terrain_config.yaml"
    print(f"{os.path.dirname(os.path.realpath(__file__))}/{config_file}")
    with open(f"{os.path.dirname(os.path.realpath(__file__))}/{config_file}", "r", encoding="utf-8") as f:
        terrain_config = yaml.load(f, Loader=yaml.FullLoader)

    # 使用get嵌套读取
    print(terrain_config["x"])
    print(terrain_config.get("x"))
    print(terrain_config.get("plum_blossom").get("foot_clearance_m"))
    print(terrain_config.get(('plum_blossom', 'foot_clearance_m'), 0.04))
    print(terrain_config.get('1', 0.04))
    print(terrain_config.get(('1', '2'), 0.04))

    exit()

    model = mujoco.MjModel.from_xml_path(f"{os.path.dirname(os.path.realpath(__file__))}/robots/go2/scene_terrain.xml")
    data = mujoco.MjData(model)

    # 动态生成bump
    # terrain_changer = TerrainChanger(model, data, action_dims={'bump':4}, config_file="terrain_config.yaml")
    # terrain_changer.run()

    # 三角函数组合
    # terrain_changer = TerrainChanger(model, data, action_dims={}, config_file="terrain_config.yaml")
    # angle_array = []
    # for i in range(10):
    #     angle_array.append([])
    #     for j in range(10):
    #         angle_array[i].append([np.random.uniform(0, 2 * np.pi), np.random.uniform(-1, 1)])
    # terrain_changer.generate_trig_terrain(angle_array)
    # terrain_changer.enforce_safe_spawn_area()
    # terrain_changer.run()

    # 初始化bumps
    terrain_changer = TerrainChanger(model, data, action_dims={}, config_file="terrain_config.yaml")
    bumps_array = []
    for _ in range(100):
        bumps_array.append([np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(-0.2, 0.2), np.random.uniform(0.1, 5)])
    terrain_changer.generate_bumps_terrain(bumps_array)
    terrain_changer.run()
