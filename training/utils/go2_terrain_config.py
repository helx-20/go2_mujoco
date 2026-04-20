import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from training.utils.legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class GO2TerrainCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0, 0, 0.42] # x,y,z [m] z =0.42 # -3.0, 1.5, 0.7
        # pos = [-0.0, 0.0, 0.42] # x,y,z [m] z =0.42 # -3.0, 1.5, 0.7
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 20.}  # [N*m/rad]
        damping = {'joint': 0.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        name = "go2"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter

    # class domain_rand( LeggedRobotCfg.domain_rand):
    #     push_robots = False
    #     randomize_friction = False
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25
        only_positive_rewards = False

        class scales( LeggedRobotCfg.rewards.scales ):
            torques = 0.0 # -0.0002
            dof_pos_limits = -0.0 # -1.0
            # termination = -1000 # -1000 # -5000 # -0.5
            tracking_lin_vel = 0.0 # 暂时加的
            tracking_ang_vel = 0.0 # 暂时加的
            termination = -0.0
            tracking_lin_vel = 0.0
            tracking_ang_vel = 0.0
            lin_vel_z = -0.0
            ang_vel_xy = -0.00
            orientation = -0.
            dof_vel = -0.
            dof_acc = 0 #-2.5e-7
            base_height = -0. 
            feet_air_time =  0 #1.0
            feet_stumble = -0.0 
            action_rate = 0 #-0.01
            stand_still = -0.
            collision = -0.0
            failed = -1.0
            success = 1.0

        # class scales( LeggedRobotCfg.rewards.scales ):
        #     torques = -0.0002
        #     dof_pos_limits = -1.0
        #     collision = -0.0
        #     failed = -10.0
        #     success = 10.0

    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = "trimesh" # "heightfield" or "trimesh" or "ground_plane"
        # num_rows = 1
        # num_cols = 1
        border_size = 25 # 25
        terrain_length = 8. # 8.
        terrain_width = 8. # 8.
        curriculum = True
        horizontal_scale = 0.1 # 0.05 # 自定义
        slope_treshold = 1.5 # 自定义

class GO2TerrainCfgPPO( LeggedRobotCfgPPO ):
    seed = -1
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'go2_terrain'
