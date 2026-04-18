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


def update_command(data, cmd, heading_stiffness, heading_target, heading_command=True):
    """Post-processes the velocity command.

    This function sets velocity command to zero for standing environments and computes angular
    velocity from heading direction if the heading_command flag is set.
    """
    if heading_command:
        current_heading = quat_to_heading_w(data.qpos[3:7])
        heading_err = wrap_to_pi(heading_target - current_heading)
        cmd[2] = np.clip(heading_err * heading_stiffness, -1, 1)
    return cmd


if __name__ == "__main__":
    # get config file name from command line
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    args = parser.parse_args()
    config_file = args.config_file
    with open(f"{os.path.dirname(os.path.realpath(__file__))}/configs/{config_file}", "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"]
        xml_path = config["xml_path"]

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        default_angles = np.array(config["default_angles"], dtype=np.float32)

        lin_vel_scale = config["lin_vel_scale"]
        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]

        cmd = np.array(config["cmd_init"], dtype=np.float32)
        heading_stiffness = config["heading_stiffness"]
        heading_target = config["heading_target"]
        heading_command = config["heading_command"]

    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # load policy
    policy = torch.jit.load(policy_path)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()

        # 一定要先等几帧，不能马上控制
        counter = 1
        while counter % control_decimation != 0:
            try:
                num_j = int(num_actions)
            except Exception:
                num_j = len(kps) if hasattr(kps, '__len__') else 12
            qpos_j = d.qpos[7:7 + num_j]
            qvel_j = d.qvel[6:6 + num_j]
            tau = pd_control(target_dof_pos, qpos_j, kps, np.zeros_like(kds), qvel_j, kds)
            d.ctrl[:] = tau
            mujoco.mj_step(m, d)
            counter += 1

        while viewer.is_running() and time.time() - start < simulation_duration:

            if counter % control_decimation == 0:
                # Apply control signal here.

                # create observation

                cmd = update_command(d, cmd, heading_stiffness, heading_target, heading_command)

                qj = d.qpos[7:]
                dqj = d.qvel[6:]
                quat = d.qpos[3:7]
                ang_vel = d.qvel[3:6]
                lin_vel = d.qvel[0:3]

                qj = (qj - default_angles) * dof_pos_scale
                dqj = dqj * dof_vel_scale
                gravity_orientation = get_gravity_orientation(quat)
                lin_vel = lin_vel * lin_vel_scale
                ang_vel = ang_vel * ang_vel_scale

                obs[0:3] = lin_vel
                obs[3:6] = ang_vel
                obs[6:9] = gravity_orientation
                obs[9:12] = cmd * cmd_scale
                obs[12:24] = qj
                obs[24:36] = dqj
                obs[36:48] = action

                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                # policy inference
                action = policy(obs_tensor).detach().numpy().squeeze()
                # transform action to target_dof_pos
                target_dof_pos = action * action_scale + default_angles

            try:
                num_j = int(num_actions)
            except Exception:
                num_j = len(kps) if hasattr(kps, '__len__') else 12
            qpos_j = d.qpos[7:7 + num_j]
            qvel_j = d.qvel[6:6 + num_j]
            tau = pd_control(target_dof_pos, qpos_j, kps, np.zeros_like(kds), qvel_j, kds)
            d.ctrl[:] = tau
            mujoco.mj_step(m, d)

            counter += 1


            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

