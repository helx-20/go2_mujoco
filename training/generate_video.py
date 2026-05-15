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


def capture_frame(renderer, data, cam, scene_option=None):
    """Render an offscreen RGB frame using mujoco.Renderer. Returns HxWx3 uint8."""
    if scene_option is not None:
        renderer.update_scene(data, camera=cam, scene_option=scene_option)
    else:
        renderer.update_scene(data, camera=cam)
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

    trainer = TestEnv(policy=pretrain_wrapper, safe_policy=safe_policy, config_file_path=config_file_path, terrain_config_file=terrain_cfg, criticality_model=criticality_model, critical_threshold=0.5, collect_training_data=False, render=True)
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
            rgb = capture_frame(renderer, trainer.data, capture_cam, scene_option=trainer.viewer.opt)
            fname = os.path.join(frames_dir, f'frame_{frame_idx:06d}.png')
            imageio.imwrite(fname, rgb)
            frame_idx += 1
        except Exception as e:
            print(f'frame capture failed at frame {frame_idx}: {e}')

    trainer.on_terrain_update = on_terrain_update
    trainer.frame_callback = frame_callback

    try:
        for ep in range(1):
            obs, _ = env.reset()
            done = False
            step = 0
            while not done and step < args.steps:
                if action_space.shape[0] > 0:
                    a = action_space.sample()
                    action = np.asarray(a, dtype=np.float32)
                else:
                    action = np.array([], dtype=np.float32)

                _, _, terminated, truncated, _ = env.step(action)
                done = bool(terminated) or bool(truncated)

                step += 1
                if args.delay > 0:
                    time.sleep(args.delay)

    finally:
        pass

    # compose the per-frame PNGs into mp4 with fps chosen so total length == args.duration
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if frames_dir and frame_idx > 0:
        compose_fps = max(1, int(round(frame_idx / max(args.duration, 1e-3))))
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
    parser.add_argument('--controller_path', type=str, default='training/models/run_offline_round5/best.policy.pt')
    parser.add_argument('--out', default='training/videos/rollout.mp4', help='output mp4 path')
    parser.add_argument('--steps', type=int, default=800, help='max frames to record')
    parser.add_argument('--max_steps', type=int, default=1000, help='env max steps per episode')
    parser.add_argument('--fps', type=int, default=60, help='unused for compose; kept for compatibility')
    parser.add_argument('--duration', type=float, default=8.0, help='target output video length in seconds')
    parser.add_argument('--delay', type=float, default=0.0, help='sleep seconds between steps (for realtime)')
    parser.add_argument('--width', type=int, default=1200, help='frame width for offscreen render')
    parser.add_argument('--height', type=int, default=800, help='frame height for offscreen render')
    parser.add_argument('--viewer', action='store_true', help='force using MuJoCo viewer for capture (use under xvfb-run)')
    parser.add_argument('--frames-dir', type=str, default="training/videos/frames", help='directory to save per-frame PNGs and compose with ffmpeg after run')
    parser.add_argument('--grab-x', type=int, default=0, help='x offset for X11 grab (use to skip left toolbar)')
    parser.add_argument('--grab-y', type=int, default=0, help='y offset for X11 grab')
    parser.add_argument('--grab-width', type=int, default=None, help='width for X11 grab (defaults to --width)')
    parser.add_argument('--grab-height', type=int, default=None, help='height for X11 grab (defaults to --height)')
    args = parser.parse_args()
    main(args)
