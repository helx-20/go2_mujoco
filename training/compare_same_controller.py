import csv
import json
import numpy as np
import os
import traceback
import mujoco
# training/compare_same_controller.py
import os
import sys
import argparse
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from training.utils.test_env import TestEnv, TerrainGymEnv
from training.controller_env import ControllerEnv
from stable_baselines3 import PPO
from deploy_mujoco.utils import pd_control, quat_to_rpy


def safe_get(d, keys, default=None):
    for k in keys:
        if isinstance(d, dict) and k in d:
            return d[k]
    return default


def run_compare(n_eps=5, out_dir="/tmp/compare_trace", last_n=12):
    os.makedirs(out_dir, exist_ok=True)

    policy = PPO.load('/home/linxuan/Embodied/go2_mujoco/training/model/actor_init.zip', device='cpu').policy

    class PolicyOnlyWrapper(torch.nn.Module):
        def __init__(self, net_pi, act_net):
            super().__init__()
            self.net_pi = net_pi
            self.act_net = act_net

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            latent = self.net_pi(x)
            actions = self.act_net(latent)
            return actions

    policy_net = policy.mlp_extractor.policy_net
    action_net = policy.action_net
    policy_net.to('cpu')
    action_net.to('cpu')
    policy_net.eval()
    action_net.eval()

    wrapper = PolicyOnlyWrapper(policy_net, action_net).cpu()

    trainer = TestEnv(policy=wrapper, config_file_path="go2_training.yaml", terrain_config_file="terrain_config.yaml")
    t_env = TerrainGymEnv(trainer, max_episode_steps=1000)
    c_env = ControllerEnv(warmup_policy=wrapper, config_file_path="go2_training.yaml", max_episode_steps=1000)
    # enable optional per-mj-step tracing in ControllerEnv for early steps
    try:
        c_env.simtrace_enable = True
        c_env.simtrace_out_dir = out_dir
        c_env.simtrace_limit_steps = 3
        c_env.simtrace_ep = 0
    except Exception:
        pass

    # terrain action vector zeros (apply same terrain change to both envs)
    try:
        ta_dim = getattr(trainer, 'total_action_dims', None) or getattr(trainer.terrain_changer, 'action_dim', None)
        terrain_action = np.zeros(int(ta_dim)) if ta_dim is not None else np.zeros(1)
    except Exception:
        terrain_action = np.zeros(1)

    stats = {'controller_env': {'stuck': 0, 'fallen': 0, 'other': 0}, 'test_env': {'stuck': 0, 'fallen': 0, 'other': 0}}

    import argparse as _arg
    parser = _arg.ArgumentParser(add_help=False)
    parser.add_argument('--inject-trainer-tau', action='store_true', help='Apply Trainer tau into ControllerEnv each step for debugging')
    parser.add_argument('--sync-state', action='store_true', help='Copy Trainer state (qpos/qvel/ctrl) into ControllerEnv before each step')
    _args, _ = parser.parse_known_args()
    inject_trainer_tau = bool(getattr(_args, 'inject_trainer_tau', False))
    sync_state = bool(getattr(_args, 'sync_state', False))

    # when injecting trainer tau, optionally force it on every muJoCo substep
    try:
        c_env.force_trainer_tau_always = inject_trainer_tau
    except Exception:
        pass

    for ep in range(n_eps):
        t_obs, t_info = t_env.reset()
        c_obs, c_info = c_env.reset()

        # DEBUG HACK: copy trainer's full state into controller env to enforce
        # identical starting conditions. This helps determine whether early
        # divergence is due to different resets/terrain or due to mapping/apply
        # logic inside ControllerEnv.
        try:
            c_env.data.qpos[:] = trainer.data.qpos[:]
            c_env.data.qvel[:] = trainer.data.qvel[:]
            c_env.data.ctrl[:] = trainer.data.ctrl[:]
            try:
                mujoco.mj_forward(c_env.model, c_env.data)
            except Exception:
                pass
        except Exception:
            pass

        # keep in-memory per-step records for possible extraction
        rows_buffer = []

        # Warmup ControllerEnv to match TestEnv's protective warmup so initial base speed/state align
        try:
            total_sim_steps = int(getattr(trainer, 'init_skip_time', 2) / float(c_env.model.opt.timestep)) + int(getattr(trainer, 'init_skip_frame', 10))
        except Exception:
            total_sim_steps = int(getattr(trainer, 'control_decimation', 10) * 10)
        try:
            target_dof_pos = trainer.go2_controller.default_angles.copy()
        except Exception:
            target_dof_pos = np.zeros(getattr(c_env, 'num_actions', 12), dtype=np.float32)
        for sim_i in range(total_sim_steps):
            if sim_i >= 10 and sim_i % int(getattr(trainer, 'control_decimation', 10)) == 0:
                try:
                    target_dof_pos = trainer.go2_controller.compute_action(c_env.data)
                except Exception:
                    target_dof_pos = trainer.go2_controller.default_angles.copy()
            try:
                try:
                    num_j = int(getattr(trainer.go2_controller, 'num_actions', getattr(c_env, 'num_actions', 12)))
                except Exception:
                    num_j = 12
                qpos_j = c_env.data.qpos[7:7 + num_j]
                qvel_j = c_env.data.qvel[6:6 + num_j]
                tau = pd_control(target_dof_pos, qpos_j, trainer.go2_controller.kps, np.zeros_like(trainer.go2_controller.kds), qvel_j, trainer.go2_controller.kds)
                c_env.data.ctrl[:] = tau
                mujoco.mj_step(c_env.model, c_env.data)
            except Exception:
                break

        # After ControllerEnv warmup, re-sync full Trainer state into ControllerEnv
        # to ensure both environments start the main loop from identical physics.
        try:
            c_env.data.qpos[:] = trainer.data.qpos[:]
            c_env.data.qvel[:] = trainer.data.qvel[:]
            c_env.data.ctrl[:] = trainer.data.ctrl[:]
        except Exception:
            pass

        csv_path = os.path.join(out_dir, f"trace_ep{ep}.csv")
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            header = [
                'step', 'sim_time',
                'base_speed_t', 'base_speed_c',
                'target_dof_pos', 'action_sent',
                'qpos_joints_t', 'qpos_joints_c', 'qvel_joints_t', 'qvel_joints_c',
                'default_angles', 'action_scale',
                'tau_t_norm', 'tau_c_norm', 'tau_t', 'tau_c',
                'base_qpos_t', 'base_qpos_c', 'roll_t', 'pitch_t', 'roll_c', 'pitch_c',
                'stuck_t', 'stuck_c', 'fallen_t', 'fallen_c'
            ]
            # add trainer/controller ground sampling metadata
            header += [
                't_geom_id', 't_hfield_id', 't_center', 't_size', 't_z_scale', 't_rowcol', 't_hfield_val', 'ground_z_t',
                'c_geom_id', 'c_hfield_id', 'c_center', 'c_size', 'c_z_scale', 'c_rowcol', 'c_hfield_val', 'ground_z_c'
            ]
            writer.writerow(header)

            done_t = done_c = False
            step = 0
            post_sync_qpos = None
            post_sync_qvel = None
            while not (done_t and done_c):
                # ensure terrain synced
                try:
                    if hasattr(trainer, 'terrain_changer'):
                        trainer.terrain_changer.apply_action_vector(terrain_action)
                        trainer.terrain_changer._refresh_terrain_safe()
                    if hasattr(c_env, 'terrain_changer'):
                        c_env.terrain_changer.apply_action_vector(terrain_action)
                        c_env.terrain_changer._refresh_terrain_safe()
                except Exception:
                    pass

                # STEP ORDER CHANGE: run Trainer first to update its internal controller
                # (this mirrors TestEnv.step which applies terrain then runs the Go2 controller),
                # then construct the normalized action for ControllerEnv from Trainer's last policy output
                # (prefer action_policy_prev), then step ControllerEnv.
                try:
                    t_obs, r_t, term_t, trunc_t, info_t = t_env.step(terrain_action)
                except Exception:
                    try:
                        t_obs, r_t, done_t_f, info_t = t_env.step(terrain_action)
                        term_t = done_t_f
                        trunc_t = False
                    except Exception:
                        t_obs = None; r_t = 0.0; term_t = False; trunc_t = False; info_t = {}

                # build target/action from trainer's internal controller state
                try:
                    ctrl = trainer.go2_controller
                    # if controller recorded previous policy action, use it; else fallback to compute_action
                    if getattr(ctrl, 'action_policy_prev', None) is not None and getattr(ctrl, 'action_policy_prev').size > 0:
                        a_pol = np.asarray(ctrl.action_policy_prev, dtype=np.float32)
                        a_scale = getattr(ctrl, 'action_scale', getattr(c_env, 'action_scale', 1.0))
                        d_angles = getattr(ctrl, 'default_angles', getattr(c_env, 'default_angles', np.zeros_like(a_pol)))
                        target = a_pol * a_scale + d_angles
                    else:
                        try:
                            target = ctrl.compute_action(trainer.data)
                        except Exception:
                            target = np.asarray(getattr(ctrl, 'default_angles', []), dtype=np.float32)
                    target = np.asarray(target, dtype=np.float32)
                except Exception:
                    target = np.array(getattr(trainer.go2_controller, 'default_angles', []), dtype=np.float32)

                # normalize to ControllerEnv action space (inverse of target = action_scale*act + default)
                try:
                    default_angles = np.asarray(getattr(c_env, 'default_angles'))
                    action_scale = np.asarray(getattr(c_env, 'action_scale'))
                    denom = np.maximum(action_scale, 1e-6)
                    action = ((target - default_angles) / denom).astype(np.float32)
                except Exception:
                    action = np.asarray(target, dtype=np.float32)

                # optionally inject trainer tau into ControllerEnv or sync full state
                if inject_trainer_tau:
                    try:
                        c_env.force_tau = np.asarray(trainer.data.ctrl, dtype=np.float32).copy()
                    except Exception:
                        c_env.force_tau = None
                    if sync_state:
                        try:
                            c_env.data.qpos[:] = trainer.data.qpos[:]
                            c_env.data.qvel[:] = trainer.data.qvel[:]
                            c_env.data.ctrl[:] = trainer.data.ctrl[:]
                            try:
                                mujoco.mj_forward(c_env.model, c_env.data)
                            except Exception:
                                pass
                            # save post-sync snapshots for CSV writing
                            post_sync_qpos = np.asarray(trainer.data.qpos).copy()
                            post_sync_qvel = np.asarray(trainer.data.qvel).copy()
                            # immediate debug dump after sync to verify copying worked
                            try:
                                if step < 3:
                                    import json as _json
                                    sync_dbg = {
                                        'step': int(step),
                                        'sim_time_pre': float(safe_get(info_t, ['sim_time', 'simtime', 'time'], 0.0)),
                                        'trainer_qpos_after_step': np.asarray(trainer.data.qpos).tolist(),
                                        'controller_qpos_after_sync': np.asarray(c_env.data.qpos).tolist(),
                                        'trainer_qvel_after_step': np.asarray(trainer.data.qvel).tolist(),
                                        'controller_qvel_after_sync': np.asarray(c_env.data.qvel).tolist(),
                                    }
                                    sync_path = os.path.join(out_dir, f"debug_ep{ep}_after_sync_step{step}.json")
                                    with open(sync_path, 'w') as _sf:
                                        _json.dump(sync_dbg, _sf)
                                    # extra detailed sync dump including ground sampling and terrain params
                                    try:
                                        x = float(post_sync_qpos[0]) if post_sync_qpos is not None else float(trainer.data.qpos[0])
                                        y = float(post_sync_qpos[1]) if post_sync_qpos is not None else float(trainer.data.qpos[1])
                                    except Exception:
                                        x = y = None
                                    try:
                                        t_ground = trainer._get_ground_height_at_xy(x, y) if x is not None else None
                                    except Exception:
                                        t_ground = None
                                    try:
                                        c_ground = c_env._get_ground_height_at_xy(x, y) if x is not None else None
                                    except Exception:
                                        c_ground = None
                                    try:
                                        t_tc = getattr(trainer, 'terrain_changer', None)
                                        c_tc = getattr(c_env, 'terrain_changer', None)
                                        t_tc_info = None
                                        c_tc_info = None
                                        if t_tc is not None:
                                            t_tc_info = {
                                                'has_hfield': getattr(t_tc, 'hfield', None) is not None,
                                                'terrain_center': (getattr(t_tc, 'terrain_center_x', None), getattr(t_tc, 'terrain_center_y', None), getattr(t_tc, 'terrain_center_z', None)),
                                                'terrain_size': (getattr(t_tc, 'terrain_size_x', None), getattr(t_tc, 'terrain_size_y', None)),
                                                'terrain_z_scale': getattr(t_tc, 'terrain_z_scale', None),
                                            }
                                        if c_tc is not None:
                                            c_tc_info = {
                                                'has_hfield': getattr(c_tc, 'hfield', None) is not None,
                                                'terrain_center': (getattr(c_tc, 'terrain_center_x', None), getattr(c_tc, 'terrain_center_y', None), getattr(c_tc, 'terrain_center_z', None)),
                                                'terrain_size': (getattr(c_tc, 'terrain_size_x', None), getattr(c_tc, 'terrain_size_y', None)),
                                                'terrain_z_scale': getattr(c_tc, 'terrain_z_scale', None),
                                            }
                                    except Exception:
                                        t_tc_info = c_tc_info = None
                                    try:
                                        sync_full = {
                                            'step': int(step), 'x': x, 'y': y,
                                            'trainer_ground_z': t_ground, 'controller_ground_z': c_ground,
                                            'trainer_terrain_changer': t_tc_info, 'controller_terrain_changer': c_tc_info,
                                        }
                                        sync_full_path = os.path.join(out_dir, f"debug_ep{ep}_after_sync_full_step{step}.json")
                                        with open(sync_full_path, 'w') as _sf2:
                                            _json.dump(sync_full, _sf2)
                                    except Exception:
                                        pass
                            except Exception:
                                pass
                            # Also copy terrain hfield and sampling params if available
                            try:
                                if hasattr(trainer, 'terrain_changer') and hasattr(c_env, 'terrain_changer') and getattr(trainer.terrain_changer, 'hfield', None) is not None:
                                    try:
                                        c_env.terrain_changer.hfield = np.asarray(trainer.terrain_changer.hfield).copy()
                                    except Exception:
                                        c_env.terrain_changer.hfield = trainer.terrain_changer.hfield
                                    for attr in ['geom_id', 'hfield_geom_id', 'hfield_id',
                                                 'terrain_size_x', 'terrain_size_y', 'terrain_z_scale',
                                                 'terrain_center_x', 'terrain_center_y', 'terrain_center_z']:
                                        try:
                                            if hasattr(trainer.terrain_changer, attr):
                                                setattr(c_env.terrain_changer, attr, getattr(trainer.terrain_changer, attr))
                                        except Exception:
                                            pass
                                    # ensure model-side fallbacks reference same ids when possible
                                    try:
                                        if getattr(trainer, 'hfield_id', None) is not None:
                                            c_env.hfield_id = trainer.hfield_id
                                        if getattr(trainer, 'hfield_geom_id', None) is not None:
                                            c_env.hfield_geom_id = trainer.hfield_geom_id
                                    except Exception:
                                        pass
                            except Exception:
                                pass
                        except Exception:
                            pass

                # finally step ControllerEnv using the normalized action derived from Trainer
                try:
                    c_obs, r_c, term_c, trunc_c, info_c = c_env.step(action)
                except Exception:
                    try:
                        c_obs, r_c, done_c_f, info_c = c_env.step(action)
                        term_c = done_c_f
                        trunc_c = False
                    except Exception:
                        c_obs = None; r_c = 0.0; term_c = False; trunc_c = False; info_c = {}

                sim_time = safe_get(info_t, ['sim_time', 'simtime', 'time'], safe_get(info_c, ['sim_time', 'simtime', 'time'], step * 0.02))
                base_speed_t = safe_get(info_t, ['speed', 'base_speed', 'base_lin_speed', 'base_lin_vel'], None)
                base_speed_c = safe_get(info_c, ['speed', 'base_speed', 'base_lin_speed', 'base_lin_vel'], None)

                # collect ctrl (tau) and base pose info for both envs
                try:
                    tau_t = np.asarray(trainer.data.ctrl, dtype=np.float32).tolist()
                    tau_t_norm = float(np.linalg.norm(np.asarray(trainer.data.ctrl, dtype=np.float32)))
                except Exception:
                    tau_t = []
                    tau_t_norm = 0.0
                try:
                    tau_c = np.asarray(c_env.data.ctrl, dtype=np.float32).tolist()
                    tau_c_norm = float(np.linalg.norm(np.asarray(c_env.data.ctrl, dtype=np.float32)))
                except Exception:
                    tau_c = []
                    tau_c_norm = 0.0

                # prefer post-sync snapshot for CSV when available
                try:
                    if post_sync_qpos is not None:
                        qpos_t_arr = post_sync_qpos[:3]
                        qpos_c_arr = post_sync_qpos[:3]
                        quat_t = post_sync_qpos[3:7]
                        quat_c = post_sync_qpos[3:7]
                    else:
                        qpos_t_arr = np.asarray(trainer.data.qpos[:3])
                        qpos_c_arr = np.asarray(c_env.data.qpos[:3])
                        quat_t = trainer.data.qpos[3:7]
                        quat_c = c_env.data.qpos[3:7]
                    qpos_t = np.asarray(qpos_t_arr, dtype=np.float32).tolist()
                    roll_t, pitch_t, _ = quat_to_rpy(quat_t)
                    qpos_c = np.asarray(qpos_c_arr, dtype=np.float32).tolist()
                    roll_c, pitch_c, _ = quat_to_rpy(quat_c)
                except Exception:
                    qpos_t = []
                    qpos_c = []
                    roll_t = pitch_t = roll_c = pitch_c = 0.0

                # joint positions/velocities (trainer vs controller) — prefer post-sync
                try:
                    if post_sync_qpos is not None:
                        qj_t = np.asarray(post_sync_qpos[7:], dtype=np.float32).tolist()
                        qj_c = np.asarray(post_sync_qpos[7:], dtype=np.float32).tolist()
                    else:
                        qj_t = np.asarray(trainer.data.qpos[7:], dtype=np.float32).tolist()
                        qj_c = np.asarray(c_env.data.qpos[7:], dtype=np.float32).tolist()
                except Exception:
                    qj_t = []
                    qj_c = []
                try:
                    if post_sync_qvel is not None:
                        dqj_t = np.asarray(post_sync_qvel[6:], dtype=np.float32).tolist()
                        dqj_c = np.asarray(post_sync_qvel[6:], dtype=np.float32).tolist()
                    else:
                        dqj_t = np.asarray(trainer.data.qvel[6:], dtype=np.float32).tolist()
                        dqj_c = np.asarray(c_env.data.qvel[6:], dtype=np.float32).tolist()
                except Exception:
                    dqj_t = []
                    dqj_c = []

                stuck_t = bool(safe_get(info_t, ['stuck'], False))
                stuck_c = bool(safe_get(info_c, ['stuck'], False))
                fallen_t = bool(safe_get(info_t, ['fallen'], False))
                fallen_c = bool(safe_get(info_c, ['fallen'], False))

                # collect ground sampling metadata for trainer and controller
                def collect_ground_meta(env, qpos_arr=None):
                    meta = {
                        'geom_id': None, 'hfield_id': None, 'center': None, 'size': None, 'z_scale': None,
                        'rowcol': None, 'hfield_val': None, 'ground_z': None
                    }
                    try:
                        q = qpos_arr if qpos_arr is not None else getattr(env, 'data').qpos
                        x = float(q[0]); y = float(q[1])
                        # try terrain_changer.hfield path
                        tc = getattr(env, 'terrain_changer', None)
                        if tc is not None and getattr(tc, 'hfield', None) is not None:
                            hfield = tc.hfield
                            geom_id = getattr(tc, 'geom_id', getattr(tc, 'hfield_geom_id', None))
                            meta['geom_id'] = int(geom_id) if geom_id is not None else None
                            nrow = int(hfield.shape[0]); ncol = int(hfield.shape[1])
                            size_x = float(getattr(tc, 'terrain_size_x', None)) if getattr(tc, 'terrain_size_x', None) is not None else None
                            size_y = float(getattr(tc, 'terrain_size_y', None)) if getattr(tc, 'terrain_size_y', None) is not None else None
                            z_scale = float(getattr(tc, 'terrain_z_scale', 1.0))
                            center_x = float(getattr(tc, 'terrain_center_x', None)) if getattr(tc, 'terrain_center_x', None) is not None else None
                            center_y = float(getattr(tc, 'terrain_center_y', None)) if getattr(tc, 'terrain_center_y', None) is not None else None
                            center_z = float(getattr(tc, 'terrain_center_z', None)) if getattr(tc, 'terrain_center_z', None) is not None else None
                            # fallback to model values if needed
                            if size_x is None or size_y is None or center_x is None or center_y is None or center_z is None:
                                try:
                                    hid = getattr(env, 'hfield_id', None)
                                    geom = getattr(env, 'hfield_geom_id', None)
                                    if hid is not None:
                                        size_x = size_x if size_x is not None else float(env.model.hfield_size[hid][0])
                                        size_y = size_y if size_y is not None else float(env.model.hfield_size[hid][1])
                                        z_scale = z_scale if z_scale is not None else float(env.model.hfield_size[hid][2])
                                    if geom is not None:
                                        center_x = center_x if center_x is not None else float(env.model.geom_pos[geom][0])
                                        center_y = center_y if center_y is not None else float(env.model.geom_pos[geom][1])
                                        center_z = center_z if center_z is not None else float(env.model.geom_pos[geom][2])
                                except Exception:
                                    pass
                            # compute uv -> row/col
                            if center_x is None or size_x is None:
                                return meta
                            u = (x - center_x + size_x) / (2.0 * size_x)
                            v = (y - center_y + size_y) / (2.0 * size_y)
                            u = float(np.clip(u, 0.0, 1.0)); v = float(np.clip(v, 0.0, 1.0))
                            col = int(np.clip(round(u * (ncol - 1)), 0, ncol - 1))
                            row = int(np.clip(round(v * (nrow - 1)), 0, nrow - 1))
                            hval = float(hfield[row, col])
                            ground_z = float(center_z + z_scale * hval)
                            meta.update({'hfield_id': None, 'center': (center_x, center_y, center_z), 'size': (size_x, size_y), 'z_scale': z_scale, 'rowcol': (int(row), int(col)), 'hfield_val': hval, 'ground_z': ground_z})
                            return meta

                        # fallback to model hfield
                        hid = getattr(env, 'hfield_id', None)
                        geom = getattr(env, 'hfield_geom_id', None)
                        if hid is None or geom is None:
                            return meta
                        hfield_id = int(hid)
                        nrow = int(env.model.hfield_nrow[hfield_id])
                        ncol = int(env.model.hfield_ncol[hfield_id])
                        size_x = float(env.model.hfield_size[hfield_id][0])
                        size_y = float(env.model.hfield_size[hfield_id][1])
                        z_scale = float(env.model.hfield_size[hfield_id][2])
                        center_x = float(env.model.geom_pos[geom][0])
                        center_y = float(env.model.geom_pos[geom][1])
                        center_z = float(env.model.geom_pos[geom][2])
                        u = (x - center_x + size_x) / (2.0 * size_x)
                        v = (y - center_y + size_y) / (2.0 * size_y)
                        u = float(np.clip(u, 0.0, 1.0)); v = float(np.clip(v, 0.0, 1.0))
                        col = int(np.clip(round(u * (ncol - 1)), 0, ncol - 1))
                        row = int(np.clip(round(v * (nrow - 1)), 0, nrow - 1))
                        try:
                            hval = float(env.model.hfield_data[row * ncol + col])
                        except Exception:
                            hval = 0.0
                        ground_z = float(center_z + z_scale * hval)
                        meta.update({'geom_id': int(geom), 'hfield_id': int(hfield_id), 'center': (center_x, center_y, center_z), 'size': (size_x, size_y), 'z_scale': z_scale, 'rowcol': (int(row), int(col)), 'hfield_val': hval, 'ground_z': ground_z})
                        return meta
                    except Exception:
                        return meta

                # choose qpos snapshot for sampling (prefer post-sync)
                qpos_for_sample = post_sync_qpos if post_sync_qpos is not None else np.asarray(trainer.data.qpos)
                t_meta = collect_ground_meta(trainer, qpos_for_sample)
                qpos_for_sample_c = post_sync_qpos if post_sync_qpos is not None else np.asarray(c_env.data.qpos)
                c_meta = collect_ground_meta(c_env, qpos_for_sample_c)

                row_vals = [
                    step, sim_time,
                    base_speed_t, base_speed_c,
                    json.dumps(np.array(target).tolist()), json.dumps(np.array(action).tolist()),
                    json.dumps(qj_t), json.dumps(qj_c), json.dumps(dqj_t), json.dumps(dqj_c),
                    json.dumps(np.array(default_angles).tolist()) if default_angles is not None else '',
                    json.dumps(np.array(action_scale).tolist()) if action_scale is not None else '',
                    tau_t_norm, tau_c_norm, json.dumps(tau_t), json.dumps(tau_c),
                    json.dumps(qpos_t), json.dumps(qpos_c), float(roll_t), float(pitch_t), float(roll_c), float(pitch_c),
                    int(stuck_t), int(stuck_c), int(fallen_t), int(fallen_c)
                ]

                # append trainer meta
                row_vals += [
                    t_meta.get('geom_id'), t_meta.get('hfield_id'), json.dumps(t_meta.get('center')), json.dumps(t_meta.get('size')), t_meta.get('z_scale'), json.dumps(t_meta.get('rowcol')), t_meta.get('hfield_val'), t_meta.get('ground_z')
                ]
                # append controller meta
                row_vals += [
                    c_meta.get('geom_id'), c_meta.get('hfield_id'), json.dumps(c_meta.get('center')), json.dumps(c_meta.get('size')), c_meta.get('z_scale'), json.dumps(c_meta.get('rowcol')), c_meta.get('hfield_val'), c_meta.get('ground_z')
                ]

                writer.writerow(row_vals)

                # Debug dump: export trainer/controller low-level state for first steps
                try:
                    if step < 3:
                        import json as _json, pathlib as _pth
                        # compute base and ground heights and rel values for both envs
                        try:
                            t_base_z = float(trainer.data.qpos[2])
                        except Exception:
                            t_base_z = None
                        try:
                            c_base_z = float(c_env.data.qpos[2])
                        except Exception:
                            c_base_z = None
                        try:
                            t_ground_z = trainer._get_ground_height_at_xy(float(trainer.data.qpos[0]), float(trainer.data.qpos[1]))
                        except Exception:
                            t_ground_z = None
                        try:
                            c_ground_z = c_env._get_ground_height_at_xy(float(c_env.data.qpos[0]), float(c_env.data.qpos[1]))
                        except Exception:
                            c_ground_z = None
                        try:
                            t_base_rel = None if t_base_z is None or t_ground_z is None else float(t_base_z - t_ground_z)
                        except Exception:
                            t_base_rel = None
                        try:
                            c_base_rel = None if c_base_z is None or c_ground_z is None else float(c_base_z - c_ground_z)
                        except Exception:
                            c_base_rel = None

                        dbg = {
                            'step': int(step),
                            'sim_time': float(sim_time),
                            'trainer_qpos': np.asarray(trainer.data.qpos).tolist(),
                            'controller_qpos': np.asarray(c_env.data.qpos).tolist(),
                            'trainer_qvel': np.asarray(trainer.data.qvel).tolist(),
                            'controller_qvel': np.asarray(c_env.data.qvel).tolist(),
                            'trainer_ctrl': np.asarray(trainer.data.ctrl).tolist(),
                            'controller_ctrl': np.asarray(c_env.data.ctrl).tolist(),
                            'trainer_base_z': t_base_z,
                            'trainer_ground_z': t_ground_z,
                            'trainer_base_rel': t_base_rel,
                            'trainer_meta': t_meta,
                            'controller_base_z': c_base_z,
                            'controller_ground_z': c_ground_z,
                            'controller_base_rel': c_base_rel,
                            'controller_meta': c_meta,
                            'trainer_info': info_t,
                            'controller_info': info_c,
                        }
                        dbg_path = os.path.join(out_dir, f"debug_ep{ep}_step{step}.json")
                        with open(dbg_path, 'w') as _jf:
                            _json.dump(dbg, _jf)
                except Exception:
                    pass

                # append structured row to buffer for later extraction if episode ends with stuck
                rows_buffer.append({
                    'step': int(step), 'sim_time': float(sim_time),
                    'base_speed_t': base_speed_t, 'base_speed_c': base_speed_c,
                    'target_dof_pos': np.array(target).tolist(), 'action_sent': np.array(action).tolist(),
                    'qpos_joints_t': qj_t, 'qpos_joints_c': qj_c, 'qvel_joints_t': dqj_t, 'qvel_joints_c': dqj_c,
                    'default_angles': np.array(default_angles).tolist() if default_angles is not None else None,
                    'action_scale': np.array(action_scale).tolist() if action_scale is not None else None,
                    'tau_t_norm': float(tau_t_norm), 'tau_c_norm': float(tau_c_norm), 'tau_t': tau_t, 'tau_c': tau_c,
                    'qpos_t': qpos_t, 'qpos_c': qpos_c, 'roll_t': float(roll_t), 'pitch_t': float(pitch_t), 'roll_c': float(roll_c), 'pitch_c': float(pitch_c),
                    'stuck_t': int(stuck_t), 'stuck_c': int(stuck_c), 'fallen_t': int(fallen_t), 'fallen_c': int(fallen_c),
                    't_meta': t_meta, 'c_meta': c_meta
                })

                step += 1

                if term_c or trunc_c:
                    key = 'stuck' if stuck_c else 'fallen' if fallen_c else 'other'
                    stats['controller_env'][key] += 1
                    done_c = True
                    # if controller env ended due to stuck, export last N steps JSON for inspection
                    if stuck_c:
                        try:
                            import pathlib
                            json_path = os.path.join(out_dir, f"trace_ep{ep}.stuck_last{last_n}.json")
                            last_rows = rows_buffer[-int(last_n):] if len(rows_buffer) >= int(last_n) else rows_buffer
                            with open(json_path, 'w') as jf:
                                json.dump(last_rows, jf, indent=2)
                            print(f"Wrote stuck extract for ep {ep} -> {json_path}")
                        except Exception:
                            pass
                if term_t or trunc_t:
                    key = 'stuck' if stuck_t else 'fallen' if fallen_t else 'other'
                    stats['test_env'][key] += 1
                    done_t = True

                # safety
                if step > 2000:
                    break

        print(f"Wrote trace for ep {ep} -> {csv_path}")
        print('stats:', stats)

    print('stats:', stats)


if __name__ == '__main__':
    try:
        run_compare(n_eps=1)
    except Exception:
        traceback.print_exc()
        raise