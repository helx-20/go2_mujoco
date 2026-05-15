#!/usr/bin/env python3
"""
Record a single rollout from the Go2 MuJoCo environment and save as a video.

This script will try several methods to capture frames from the MuJoCo viewer:
  - call `viewer.render()` if available
  - call `viewer.read_pixels(width, height)` if available
If neither works the script will print an error and exit.

Usage example:
  xvfb-run --auto-servernum -s "-screen 0 1200x800x24 -ac -noreset" python training/generate_video.py --viewer --duration 8
  (pkill -TERM Xvfb; pkill -TERM -f xvfb-run; sleep 1; pkill -KILL Xvfb; pkill -KILL -f xvfb-run) &>/dev/null || true

Note: run this from the repository root so imports resolve correctly.
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import argparse
import numpy as np
import imageio
import time
import subprocess
import shutil
import os

import mujoco
import torch

def capture_frame(renderer, data, cam, scene_option=None, ground_color=None):
    """Render an offscreen RGB frame using mujoco.Renderer. Returns HxWx3 uint8."""
    if scene_option is not None:
        renderer.update_scene(data, camera=cam, scene_option=scene_option)
    else:
        renderer.update_scene(data, camera=cam)

    # disable wireframe overlay (grid lines) and let caller override ground color
    try:
        # Turn off wireframe render flag so mesh/grid lines aren't shown
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_WIREFRAME] = False
    except Exception:
        pass

    # optionally paint the ground (hfield/plane) with a solid color
    if ground_color is not None:
        try:
            # iterate visible geoms and override ground color
            for i in range(getattr(renderer.scene, 'ngeom', 0)):
                g = renderer.scene.geoms[i]
                if int(getattr(g, 'type', -1)) in (mujoco.mjtGeom.mjGEOM_HFIELD, mujoco.mjtGeom.mjGEOM_PLANE):
                    try:
                        g.rgba[0] = float(ground_color[0])
                        g.rgba[1] = float(ground_color[1])
                        g.rgba[2] = float(ground_color[2])
                        g.rgba[3] = 1.0
                        # remove textured material if present
                        g.matid = -1
                    except Exception:
                        pass
        except Exception:
            pass

    return renderer.render()


def sync_camera(dst, src):
    """Copy camera fields from src (viewer.cam) to dst (MjvCamera)."""
    dst.type = src.type
    dst.fixedcamid = src.fixedcamid
    dst.trackbodyid = src.trackbodyid
    dst.azimuth = src.azimuth
    dst.elevation = src.elevation
    dst.distance = src.distance
    dst.lookat[:] = src.lookat[:]


def make_camera_from_viewer(viewer):
    """Build a MjvCamera that mirrors the passive viewer's current camera."""
    cam = mujoco.MjvCamera()
    sync_camera(cam, viewer.cam)
    return cam


def detect_viewer_geometry(display, title_keywords=None):
    """Try to detect the viewer window geometry (x,y,width,height) using X11 tools.

    Tries `xdotool` then `wmctrl`. Returns (x,y,w,h) or None.
    """
    if title_keywords is None:
        title_keywords = ['MuJoCo', 'mujoco', 'Mujoco', 'MJVIEWER']

    env = os.environ.copy()
    if display:
        env['DISPLAY'] = display

    # try xdotool
    xdotool = shutil.which('xdotool')
    if xdotool:
        for kw in title_keywords:
            try:
                res = subprocess.run([xdotool, 'search', '--name', kw], capture_output=True, env=env, text=True)
                if res.returncode == 0 and res.stdout.strip():
                    # take first id
                    win = res.stdout.strip().splitlines()[0]
                    geom = subprocess.run([xdotool, 'getwindowgeometry', '--shell', win], capture_output=True, env=env, text=True)
                    if geom.returncode == 0:
                        lines = geom.stdout.splitlines()
                        vals = {k: int(v) for k, v in (l.split('=') for l in lines if '=' in l)}
                        x = vals.get('X')
                        y = vals.get('Y')
                        w = vals.get('WIDTH')
                        h = vals.get('HEIGHT')
                        if None not in (x, y, w, h):
                            return x, y, w, h
            except Exception:
                pass

    # try wmctrl
    wmctrl = shutil.which('wmctrl')
    if wmctrl:
        try:
            res = subprocess.run([wmctrl, '-lG'], capture_output=True, env=env, text=True)
            if res.returncode == 0 and res.stdout:
                for line in res.stdout.splitlines():
                    parts = line.split(None, 7)
                    if len(parts) >= 8:
                        # parts: win_id, desktop, x, y, w, h, host, title
                        title = parts[7]
                        for kw in title_keywords:
                            if kw in title:
                                x = int(parts[2]); y = int(parts[3]); w = int(parts[4]); h = int(parts[5])
                                return x, y, w, h
        except Exception:
            pass

    return None


def main(args):
    config_file_path = "go2_training.yaml"
    terrain_cfg = "terrain_config.yaml"

    # Import environment and RL libraries here (after MUJOCO_GL is set)
    from training.utils.render_env import TestEnv, TerrainGymEnv
    from training.utils.train_env import TrainEnv
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3 import PPO
    import torch

    class PolicyOnlyWrapper(torch.nn.Module):
        def __init__(self, net_pi, act_net):
            super().__init__()
            self.net_pi = net_pi
            self.act_net = act_net

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            latent = self.net_pi(x)
            actions = self.act_net(latent)
            return actions

    # Optional: load criticality model if present
    try:
        from criticality.utils.criticality_model import SimpleClassifier
        criticality_model = SimpleClassifier(input_dim=56)
        criticality_model.load_state_dict(torch.load('criticality/stage1_plus/model/stage1_plus_criticality_best_new_3.pt', map_location='cpu', weights_only=False))
        criticality_model.to('cpu').eval()
    except Exception:
        criticality_model = None

    # Pretrain wrapper (best effort, ignore failures)
    pretrain_wrapper = None
    try:
        from stable_baselines3 import PPO as SB3PPO
        sb3_pretrain_model = SB3PPO.load('training/models/actor_init.zip', device='cpu')
        pretrain_wrapper = PolicyOnlyWrapper(sb3_pretrain_model.policy.mlp_extractor.policy_net.to('cpu').eval(), sb3_pretrain_model.policy.action_net.to('cpu').eval()).cpu()
    except Exception:
        pretrain_wrapper = None

    # Load controller (support .zip (SB3) or .pt state dict)
    controller_path = args.controller_path
    if controller_path is None:
        print('Please provide --controller_path')
        return

    # helper similar to test_model.py
    def make_env_fn(trainer, max_episode_steps=1000):
        def _thunk():
            env = TrainEnv(trainer=trainer, max_episode_steps=max_episode_steps)
            return Monitor(env)

        return _thunk

    sb3 = None
    try:
        if controller_path.endswith('.zip'):
            sb3 = PPO.load(controller_path, device='cpu')
        elif controller_path.endswith('.pt'):
            hidden_sizes = [512, 256, 128]
            policy_kwargs = {'net_arch': dict(pi=hidden_sizes, vf=hidden_sizes), 'activation_fn': torch.nn.ELU}
            trainer_tmp = TestEnv(policy=None, config_file_path=config_file_path, terrain_config_file=terrain_cfg)
            dummy_env = make_env_fn(trainer_tmp, max_episode_steps=1000)()
            sb3 = PPO('MlpPolicy', dummy_env, policy_kwargs=policy_kwargs, device='cpu')
            state_dict = torch.load(controller_path, map_location='cpu', weights_only=False)
            # some checkpoints store under 'policy_state_dict'
            if isinstance(state_dict, dict) and 'policy_state_dict' in state_dict:
                state_dict = state_dict['policy_state_dict']
            sb3.policy.load_state_dict(state_dict)
        else:
            print('Unsupported controller_path format')
            return
    except Exception as e:
        print('Failed to load controller:', e)
        return

    policy_net = sb3.policy.mlp_extractor.policy_net
    action_net = sb3.policy.action_net
    policy_net.to('cpu')
    action_net.to('cpu')
    policy_net.eval()
    action_net.eval()
    safe_policy = PolicyOnlyWrapper(policy_net, action_net).cpu()

    trainer = TestEnv(policy=pretrain_wrapper, safe_policy=safe_policy, config_file_path=config_file_path, terrain_config_file=terrain_cfg, criticality_model=criticality_model, critical_threshold=args.critical_threshold, collect_training_data=False, render=True)
    env = TerrainGymEnv(trainer, max_episode_steps=args.max_steps)

    width = args.width
    height = args.height
    if args.viewer:
        try:
            # ensure render flag is enabled so start_viewer actually launches
            trainer.render = True
            trainer.start_viewer()
            print('Started MuJoCo viewer for capture (user requested --viewer)')
        except Exception as e2:
            print('Failed to start viewer despite --viewer:', e2)
            print('Exiting to avoid unstable native crashes. Try setting MUJOCO_GL=osmesa or installing offscreen support.')
            return
    else:
        print('No offscreen rendering available and --viewer not set; exiting.')
        print('Options: run with `--viewer` under xvfb-run, or install offscreen support and set MUJOCO_GL=osmesa/egl.')
        return

    # Bump the model's offscreen framebuffer so it fits the requested resolution.
    trainer.model.vis.global_.offwidth = max(width, int(trainer.model.vis.global_.offwidth))
    trainer.model.vis.global_.offheight = max(height, int(trainer.model.vis.global_.offheight))

    renderer = mujoco.Renderer(trainer.model, height=height, width=width)
    capture_cam = make_camera_from_viewer(trainer.viewer)

    action_space = env.action_space

    frame_idx = 0
    frames_dir = args.frames_dir if hasattr(args, 'frames_dir') else None
    if frames_dir:
        os.makedirs(frames_dir, exist_ok=True)

    # parse ground color argument (r,g,b in 0..1)
    ground_color = None
    if hasattr(args, 'ground_color') and args.ground_color:
        try:
            parts = [p for p in args.ground_color.split(',') if p != '']
            if len(parts) == 3:
                ground_color = (float(parts[0]), float(parts[1]), float(parts[2]))
        except Exception:
            ground_color = None

    def on_terrain_update():
        try:
            renderer._gl_context.make_current()
            mujoco.mjr_uploadHField(trainer.model, renderer._mjr_context, trainer.terrain_changer.hfield_id)
        except Exception as e:
            print(f'hfield upload to renderer failed: {e}')

    def frame_callback():
        nonlocal frame_idx
        if not frames_dir:
            return
        try:
            sync_camera(capture_cam, trainer.viewer.cam)
            rgb = capture_frame(renderer, trainer.data, capture_cam, scene_option=trainer.viewer.opt, ground_color=ground_color)
            fname = os.path.join(frames_dir, f'frame_{frame_idx:06d}.png')
            imageio.imwrite(fname, rgb)
            frame_idx += 1
        except Exception as e:
            print(f'frame capture failed at frame {frame_idx}: {e}')

    trainer.on_terrain_update = on_terrain_update
    trainer.frame_callback = frame_callback

    action_space = env.action_space
    # prepare discretization edges for actions: 10 bins per dimension
    low = np.asarray(action_space.low, dtype=np.float32)
    high = np.asarray(action_space.high, dtype=np.float32)
    action_edges = [np.linspace(low[d], high[d], num=11) for d in range(low.shape[0])]
    D = 4
    grids = np.meshgrid(*[np.arange(10) for _ in range(D)], indexing='ij')
    bins_flat = np.stack([g.reshape(-1) for g in grids], axis=1).astype(np.int64)
    num_actions = bins_flat.shape[0]
    centers = np.zeros((num_actions, D), dtype=np.float32)
    for d in range(D):
        e = action_edges[d]
        b_idx = bins_flat[:, d]
        centers[:, d] = 0.5 * (e[b_idx] + e[b_idx + 1])
    candidates_arr = centers

    try:
        for ep in range(1):
            obs, _ = env.reset()
            done = False
            step = 0
            total_weight = 1.0
            if args.replay_data_path is not None:
                try:
                    replay_data = np.load(args.replay_data_path, allow_pickle=True)[args.replay_idx]['t_action']
                    print(f'Loaded replay data from {args.replay_data_path} (episodes={len(replay_data)})')
                except Exception as e:
                    print('Failed to load replay data:', e)
                    replay_data = []
            while not done and step < args.steps:
                with torch.no_grad():
                    t_obs = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).repeat(candidates_arr.shape[0], 1)
                    t_act = torch.from_numpy(np.asarray(candidates_arr, dtype=np.float32))
                    t_in = torch.cat([t_obs, t_act], dim=1)
                    t_out = criticality_model(t_in)
                    criticality = torch.nn.functional.softmax(t_out, dim=1)[:, 1].squeeze().cpu().numpy()

                if len(replay_data) > step:
                    idx = replay_data[step]
                    weight = 1.0
                else:
                    if not args.nade:
                        idx = np.random.randint(0, candidates_arr.shape[0])
                        weight = 1.0
                    else:
                        if np.max(criticality) > 3e-1 and total_weight > 1e-4:
                            q_list = 0.99 * (criticality / np.sum(criticality)) + 0.01 * np.ones_like(criticality) / len(criticality)
                            q_list = q_list / np.sum(q_list)
                            idx = np.random.choice(np.arange(candidates_arr.shape[0]), p=q_list)
                            # idx = int(np.argmax(criticality))
                            # weight = float((1 / len(criticality)) / (criticality[idx] / np.sum(criticality)))
                            weight = float((1 / len(criticality)) / q_list[idx])
                        else:
                            idx = np.random.randint(0, candidates_arr.shape[0])
                            weight = 1.0

                total_weight *= weight
                action = candidates_arr[idx]
                action = np.asarray(action, dtype=np.float32)
                next_obs, reward, terminated, truncated, info = env.step(action)
                obs = next_obs
                done = bool(terminated) or bool(truncated) or bool(info.get('fallen', False) or info.get('collided', False) or info.get('base_collision', False) or info.get('thigh_collision', False) or info.get('stuck', False))

                step += 1
                crash = int(bool(info.get('fallen', False) or info.get('collided', False) or info.get('base_collision', False) or info.get('thigh_collision', False) or info.get('stuck', False)))
                print(step, done, crash)
                if args.delay > 0:
                    time.sleep(args.delay)

    finally:
        pass

    # compose the per-frame PNGs into mp4 with fps chosen so total length == args.duration
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if frames_dir and frame_idx > 0:
        compose_fps = 50 # max(1, int(round(frame_idx / max(args.duration, 1e-3))))
        ffmpeg_exe = shutil.which('ffmpeg')
        if ffmpeg_exe is None:
            try:
                import imageio_ffmpeg as _iioff
                ffmpeg_exe = _iioff.get_ffmpeg_exe()
            except Exception:
                ffmpeg_exe = None
        if ffmpeg_exe is None:
            print('ffmpeg not found; PNG frames are in', frames_dir)
        else:
            cmd = [
                ffmpeg_exe, '-y',
                '-framerate', str(compose_fps),
                '-i', os.path.join(frames_dir, 'frame_%06d.png'),
                '-codec:v', 'libx264',
                '-preset', 'veryfast',
                '-pix_fmt', 'yuv420p',
                args.out,
            ]
            print(f'Composing {frame_idx} frames at {compose_fps} fps (~{args.duration:.1f}s):', ' '.join(cmd))
            subprocess.run(cmd, check=False)
            print(f'Wrote {args.out}')
    else:
        print('No frames captured; nothing to compose.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--controller_path', type=str, default='training/models/actor_init.zip')
    # parser.add_argument('--controller_path', type=str, default='training/models/run_offline_round5/best.policy.pt')
    parser.add_argument('--critical_threshold', type=float, default=0.5, help='Criticality threshold (default: 0.5)')
    parser.add_argument('--out', default='training/videos', help='output mp4 path')
    parser.add_argument('--steps', type=int, default=800, help='max frames to record')
    parser.add_argument('--max_steps', type=int, default=1000, help='env max steps per episode')
    parser.add_argument('--fps', type=int, default=60, help='unused for compose; kept for compatibility')
    parser.add_argument('--duration', type=float, default=8.0, help='target output video length in seconds')
    parser.add_argument('--delay', type=float, default=0.0, help='sleep seconds between steps (for realtime)')
    parser.add_argument('--width', type=int, default=1200, help='frame width for offscreen render')
    parser.add_argument('--height', type=int, default=800, help='frame height for offscreen render')
    parser.add_argument('--viewer', action='store_true', help='force using MuJoCo viewer for capture (use under xvfb-run)')
    parser.add_argument('--frames-dir', type=str, default=None, help='directory to save per-frame PNGs and compose with ffmpeg after run')
    parser.add_argument('--ground-color', type=str, default="0.8,0.8,0.8", help='ground color as "r,g,b" in 0..1 to paint hfield/plane geoms')
    parser.add_argument('--nade', action='store_true')
    parser.add_argument('--worker_id', type=int, default=0)
    parser.add_argument('--replay_data_path', type=str, default=None)
    parser.add_argument('--replay_idx', type=int, default=0)
    args = parser.parse_args()
    if args.controller_path.endswith('.pt'):
        args.out = os.path.join(args.out, args.controller_path.split('/')[-2], f'video_{args.worker_id}.mp4')
    elif args.controller_path.endswith('.zip'):
        args.out = os.path.join(args.out, args.controller_path.split('/')[-1][:-4], f'video_{args.worker_id}.mp4')
    args.frames_dir = os.path.join(os.path.dirname(args.out), f'frames_{args.worker_id}')
    print('Output video will be:', args.out)
    print('Frames will be saved to:', args.frames_dir)
    np.random.seed(args.worker_id)
    torch.manual_seed(args.worker_id)                  
    torch.cuda.manual_seed(args.worker_id)             
    torch.cuda.manual_seed_all(args.worker_id)          
    torch.backends.cudnn.deterministic = True
    main(args)
