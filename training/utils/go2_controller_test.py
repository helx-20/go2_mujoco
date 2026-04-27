import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

import time
import mujoco.viewer
import mujoco
import numpy as np
import torch
import yaml

from deploy_mujoco.utils import get_gravity_orientation, pd_control, quat_to_heading_w, wrap_to_pi


class Go2Controller:
    def __init__(self, config_file_path, policy):
        with open(f"{os.path.dirname(os.path.realpath(__file__))}/{config_file_path}", "r", encoding='utf-8') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        self.policy = policy

        self.num_obs = self.config["num_obs"]
        self.num_actions = self.config["num_actions"]

        self.action_policy_prev = np.zeros(self.num_actions, dtype=np.float32)

        self.xml_path = self.config["xml_path"]
        self.simulation_duration = self.config["simulation_duration"]
        self.simulation_dt = self.config["simulation_dt"]
        self.lock_camera = self.config["lock_camera"]
        self.control_decimation = self.config["control_decimation"]
        self.policy_decimation = self.config["policy_decimation"]
        self.kps = np.array(self.config["kps"], dtype=np.float32)
        self.kds = np.array(self.config["kds"], dtype=np.float32)
        self.default_angles = np.array(self.config["default_angles"], dtype=np.float32)
        self.lin_vel_scale = self.config["lin_vel_scale"]
        self.ang_vel_scale = self.config["ang_vel_scale"]
        self.dof_pos_scale = self.config["dof_pos_scale"]
        self.dof_vel_scale = self.config["dof_vel_scale"]
        self.action_scale = self.config["action_scale"]
        self.heading_stiffness = self.config["heading_stiffness"]
        self.heading_target = self.config["heading_target"]
        self.heading_command = self.config["heading_command"]
        self.cmd_scale = self.config["cmd_scale"]
        self.cmd = np.array(self.config["cmd_init"], dtype=np.float32)

    def reset(self):
        self.action_policy_prev = np.zeros(self.num_actions, dtype=np.float32)

    def get_observation(self, d):
        """Return the robot observation vector (same format used for the policy)."""
        qj = d.qpos[7:].copy()
        dqj = d.qvel[6:].copy()
        quat = d.qpos[3:7].copy()
        lin_vel = d.qvel[:3].copy()
        ang_vel = d.qvel[3:6].copy()

        qj = (qj - self.default_angles) * self.dof_pos_scale
        dqj = dqj * self.dof_vel_scale

        gravity_orientation = get_gravity_orientation(quat)
        lin_vel = lin_vel * self.lin_vel_scale
        ang_vel = ang_vel * self.ang_vel_scale

        obs = np.zeros(self.num_obs, dtype=np.float32)
        obs[:3] = lin_vel
        obs[3:6] = ang_vel
        obs[6:9] = gravity_orientation
        obs[9:12] = self.cmd * self.cmd_scale
        obs[12: 12 + self.num_actions] = qj
        obs[12 + self.num_actions: 12 + 2 * self.num_actions] = dqj
        obs[12 + 2 * self.num_actions: 12 + 3 * self.num_actions] = self.action_policy_prev

        return obs

    def get_observation_without_prev_action(self, d):
        """Return the robot observation vector (same format used for the policy)."""
        qj = d.qpos[7:].copy()
        dqj = d.qvel[6:].copy()
        quat = d.qpos[3:7].copy()
        lin_vel = d.qvel[:3].copy()
        ang_vel = d.qvel[3:6].copy()

        qj = (qj - self.default_angles) * self.dof_pos_scale
        dqj = dqj * self.dof_vel_scale

        gravity_orientation = get_gravity_orientation(quat)
        lin_vel = lin_vel * self.lin_vel_scale
        ang_vel = ang_vel * self.ang_vel_scale

        obs = np.zeros(self.num_obs - self.num_actions, dtype=np.float32)
        obs[:3] = lin_vel
        obs[3:6] = ang_vel
        obs[6:9] = gravity_orientation
        obs[9:12] = self.cmd * self.cmd_scale
        obs[12: 12 + self.num_actions] = qj
        obs[12 + self.num_actions: 12 + 2 * self.num_actions] = dqj

        return obs

    def update_command(self, data, cmd, heading_stiffness, heading_target, heading_command=True):
        """Post-processes the velocity command.

        This function sets velocity command to zero for standing environments and computes angular
        velocity from heading direction if the heading_command flag is set.
        """
        if heading_command:
            current_heading = quat_to_heading_w(data.qpos[3:7])
            heading_err = wrap_to_pi(heading_target - current_heading)
            cmd[2] = np.clip(heading_err * heading_stiffness, -1, 1)
        return cmd

    def compute_action(self, d):
        self.cmd = self.update_command(d, self.cmd, self.heading_stiffness, self.heading_target, self.heading_command)
        # create observation
        obs = self.get_observation(d)

        obs_tensor = torch.from_numpy(obs).unsqueeze(0)
        # policy inference
        action_policy = self.policy(obs_tensor).squeeze()
        if isinstance(action_policy, torch.Tensor):
            action_policy = action_policy.detach().cpu().numpy()
        
        # action_policy = np.clip(action_policy, -4, 4)
        # model action order
        target_dof_pos = action_policy * self.action_scale + self.default_angles
        # policy action order used for next step
        self.action_policy_prev[:] = action_policy

        return target_dof_pos
    
    def compute_action_with_training_data(self, d):
        self.cmd = self.update_command(d, self.cmd, self.heading_stiffness, self.heading_target, self.heading_command)
        # create observation
        obs = self.get_observation(d)

        obs_tensor = torch.from_numpy(obs).unsqueeze(0)

        # policy inference
        action_policy = self.policy(obs_tensor)
        if isinstance(action_policy, tuple):
            action_policy, value, log_prob = action_policy
            log_prob = log_prob.item()
        else:
            value = 0
            log_prob = 0
        action_policy = action_policy.squeeze()
        if isinstance(action_policy, torch.Tensor):
            action_policy = action_policy.detach().cpu().numpy()

        # action_policy = np.clip(action_policy, -4, 4)
        # model action order
        target_dof_pos = action_policy * self.action_scale + self.default_angles
        # policy action order used for next step
        self.action_policy_prev[:] = action_policy
        
        return target_dof_pos, obs, action_policy, log_prob

    def run(self):  # 独立启动仿真运行，仿真文件路径来源于config

        # Load robot model
        m = mujoco.MjModel.from_xml_path(self.xml_path)
        d = mujoco.MjData(m)
        m.opt.timestep = self.simulation_dt

        target_dof_pos = self.default_angles.copy()

        # viewer = mujoco.viewer.launch_passive(m, d)
        with mujoco.viewer.launch_passive(m, d) as viewer:
            viewer.cam.azimuth = 0
            viewer.cam.elevation = -20
            viewer.cam.distance = 1.5
            viewer.cam.lookat[:] = d.qpos[:3]
            # Close the viewer automatically after simulation_duration wall-seconds.

            # 一定要先等几帧，不能马上控制
            counter = 1
            while counter % self.control_decimation != 0:
                try:
                    num_j = int(self.num_actions)
                except Exception:
                    num_j = 12
                qpos_j = d.qpos[7:7 + num_j]
                qvel_j = d.qvel[6:6 + num_j]
                tau = pd_control(target_dof_pos, qpos_j, self.kps, np.zeros_like(self.kds), qvel_j, self.kds)
                d.ctrl[:] = tau
                mujoco.mj_step(m, d)
                counter += 1

            start = time.time()
            while viewer.is_running() and time.time() - start < self.simulation_duration:
                step_start = time.time()

                if self.lock_camera:
                    viewer.cam.lookat[:] = d.qpos[:3] # lock camera focus on the robot base

                if counter % self.control_decimation == 0:
                    target_dof_pos = self.compute_action(d)
                try:
                    num_j = int(self.num_actions)
                except Exception:
                    num_j = 12
                qpos_j = d.qpos[7:7 + num_j]
                qvel_j = d.qvel[6:6 + num_j]
                tau = pd_control(target_dof_pos, qpos_j, self.kps, np.zeros_like(self.kds), qvel_j, self.kds)
                d.ctrl[:] = tau
                counter += 1

                mujoco.mj_step(m, d)

                # Pick up changes to the physics state, apply perturbations, update options from GUI.
                viewer.sync()
                # Rudimentary time keeping, will drift relative to wall clock.
                time_until_next_step = m.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)


if __name__ == "__main__":
    config_file = "go2.yaml"

    ctrl = Go2Controller(config_file)
    ctrl.run()

