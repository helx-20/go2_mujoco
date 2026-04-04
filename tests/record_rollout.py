#!/usr/bin/env python3
"""
Record a single rollout from the Go2 MuJoCo environment and save as a video.

This script will try several methods to capture frames from the MuJoCo viewer:
  - call `viewer.render()` if available
  - call `viewer.read_pixels(width, height)` if available
If neither works the script will print an error and exit.

Usage example:
  python tests/record_rollout.py --out results/rollout.mp4 --steps 800

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

from deploy_mujoco.terrain_trainer import TerrainTrainer, TerrainGymEnv
import mujoco


def try_capture(viewer):
    """Try to capture an RGB frame from the viewer. Return ndarray HxWx3 or raise."""
    # try viewer.render()
    try:
        out = viewer.render()
        # some viewers return (rgb, depth)
        if isinstance(out, tuple) or isinstance(out, list):
            rgb = out[0]
        else:
            rgb = out
        rgb = np.asarray(rgb)
        if rgb.ndim == 3:
            return rgb
    except Exception:
        pass

    # try read_pixels
    try:
        # many viewer implementations expose read_pixels(w,h)
        w = getattr(viewer, 'width', None)
        h = getattr(viewer, 'height', None)
        if w is None or h is None:
            # fall back to common sizes
            w, h = 640, 480
        rgb = viewer.read_pixels(w, h)
        rgb = np.asarray(rgb)
        if rgb.ndim == 3:
            return rgb
    except Exception:
        pass

    raise RuntimeError('Could not capture frame from MuJoCo viewer (no supported API found)')


def safe_call(func, *args, **kwargs):
    """Call func safely; return None on exception."""
    try:
        return func(*args, **kwargs)
    except Exception:
        return None


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
    go2_cfg = ("terrain", "go2.yaml")
    terrain_cfg = "terrain_config.yaml"

    trainer = TerrainTrainer(go2_cfg, terrain_cfg)
    env = TerrainGymEnv(trainer, max_episode_steps=args.max_steps)

    # Try to create an offscreen renderer (MjRenderContextOffscreen) first.
    offscreen_ctx = None
    use_offscreen = False
    width = args.width
    height = args.height
    try:
        offscreen_ctx = mujoco.MjRenderContextOffscreen(trainer.model, 0)
        use_offscreen = True
        print('Using mujoco.MjRenderContextOffscreen for offscreen rendering')
    except Exception as e:
        print('Offscreen context not available:', e)
        # Do NOT start the viewer automatically in headless environments.
        # If the user explicitly requests viewer capture (e.g. via xvfb-run), pass --viewer.
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
    # If viewer-based capture is requested, prefer external ffmpeg X11 grab to avoid
    # calling viewer.read_pixels/render which may crash in some bindings.
    external_recorder_proc = None
    use_external_recorder = False
    pipe_proc = None
    use_pipe = False
    if args.viewer and not use_offscreen:
        display = os.environ.get('DISPLAY')
        if display:
            ffmpeg_exe = shutil.which('ffmpeg')
            if ffmpeg_exe is None:
                try:
                    import imageio_ffmpeg as _iioff
                    ffmpeg_exe = _iioff.get_ffmpeg_exe()
                except Exception:
                    ffmpeg_exe = None
            if ffmpeg_exe:
                # allow specifying grab region (offset + size) to avoid toolbars
                user_grab_x = getattr(args, 'grab_x', 0)
                user_grab_y = getattr(args, 'grab_y', 0)
                user_grab_w = getattr(args, 'grab_width', None)
                user_grab_h = getattr(args, 'grab_height', None)

                # attempt to auto-detect viewer window geometry if user did not provide offsets
                detected = None
                if (user_grab_x == 0 and user_grab_y == 0) or (user_grab_w is None or user_grab_h is None):
                    try:
                        detected = detect_viewer_geometry(display)
                    except Exception:
                        detected = None

                if detected is not None:
                    det_x, det_y, det_w, det_h = detected
                    grab_x = det_x if user_grab_x == 0 else user_grab_x
                    grab_y = det_y if user_grab_y == 0 else user_grab_y
                    grab_w = user_grab_w or det_w
                    grab_h = user_grab_h or det_h
                    print(f'Auto-detected viewer geometry: x={grab_x}, y={grab_y}, w={grab_w}, h={grab_h}')
                else:
                    grab_x = user_grab_x
                    grab_y = user_grab_y
                    grab_w = user_grab_w or width
                    grab_h = user_grab_h or height
                display_in = f"{display}+{grab_x},{grab_y}"
                cmd = [
                    ffmpeg_exe,
                    '-y',
                    '-f', 'x11grab',
                    '-video_size', f'{grab_w}x{grab_h}',
                    '-framerate', str(args.fps),
                    '-i', display_in,
                ]
                # normalize grab width/height defaults
                grab_w = grab_w or width
                grab_h = grab_h or height
                cmd += [
                    '-codec:v', 'libx264',
                    '-preset', 'veryfast',
                    '-pix_fmt', 'yuv420p',
                    args.out,
                ]
                print('Starting external ffmpeg recorder:', ' '.join(cmd))
                try:
                    external_recorder_proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    use_external_recorder = True
                except Exception as e:
                    print('Failed to start ffmpeg recorder:', e)
                    external_recorder_proc = None
                    use_external_recorder = False
                # if user requested piping raw frames and viewer exists, try pipe mode instead
                if args.pipe and hasattr(trainer, 'viewer') and trainer.viewer is not None:
                    # build ffmpeg command to accept rawvideo from stdin
                    pipe_cmd = [
                        ffmpeg_exe,
                        '-y',
                        '-f', 'rawvideo',
                        '-pix_fmt', 'rgb24',
                        '-s', f'{grab_w}x{grab_h}',
                        '-r', str(args.fps),
                        '-i', '-',
                        '-c:v', 'libx264',
                        '-preset', 'veryfast',
                        '-pix_fmt', 'yuv420p',
                        args.out,
                    ]
                    print('Starting ffmpeg pipe recorder:', ' '.join(pipe_cmd))
                    try:
                        pipe_proc = subprocess.Popen(pipe_cmd, stdin=subprocess.PIPE)
                        use_pipe = True
                        # if pipe is used, we don't need external x11grab
                        if use_external_recorder:
                            try:
                                external_recorder_proc.terminate()
                            except Exception:
                                pass
                            use_external_recorder = False
                            external_recorder_proc = None
                    except Exception as e:
                        print('Failed to start ffmpeg pipe recorder:', e)
                        pipe_proc = None
                        use_pipe = False
            else:
                print('No ffmpeg binary found (and imageio-ffmpeg not available); will attempt in-process viewer capture.')
        else:
            print('DISPLAY not set; cannot start ffmpeg X11 grab. Proceeding to in-process capture which may crash.')

    action_space = env.action_space

    frames = []
    frame_idx = 0
    frames_dir = args.frames_dir if hasattr(args, 'frames_dir') else None
    if frames_dir:
        os.makedirs(frames_dir, exist_ok=True)
    try:
        for ep in range(1):
            obs, _ = env.reset()
            done = False
            step = 0
            while not done and step < args.steps:
                frame_start = time.time()
                # sample uniform terrain action
                if action_space.shape[0] > 0:
                    a = action_space.sample()
                    action = np.asarray(a, dtype=np.float32)
                else:
                    action = np.array([], dtype=np.float32)

                next_obs, reward, terminated, truncated, info = env.step(action)
                done = bool(terminated) or bool(truncated)

                # If external ffmpeg recorder is active, force the viewer to update
                # so the X11 grab captures each simulation step. Use guarded calls
                # to avoid crashing if the binding doesn't support render().
                if use_external_recorder and hasattr(trainer, 'viewer') and trainer.viewer is not None:
                    try:
                        # sync updates camera/lookat and flips buffers in many viewers
                        safe_call(trainer.viewer.sync)
                    except Exception:
                        pass
                    try:
                        # some viewer implementations expose render(); call if available
                        safe_call(trainer.viewer.render)
                    except Exception:
                        pass

                # capture frame: prefer offscreen ctx, otherwise use viewer
                rgb = None
                if use_offscreen and offscreen_ctx is not None:
                    try:
                        # try render/read_pixels API variations
                        try:
                            offscreen_ctx.render(width, height)
                        except Exception:
                            # some bindings accept no args
                            try:
                                offscreen_ctx.render()
                            except Exception:
                                pass

                        try:
                            rgb = offscreen_ctx.read_pixels(width, height)
                        except Exception:
                            try:
                                rgb = offscreen_ctx.read_pixels()
                            except Exception:
                                rgb = None
                        if rgb is not None:
                            rgb = np.asarray(rgb)
                    except Exception as e:
                        print('Offscreen capture failed, falling back to viewer:', e)
                        rgb = None

                if rgb is None:
                    # If pipe recorder is active, try to capture frame from viewer into pipe
                    if use_pipe and hasattr(trainer, 'viewer') and trainer.viewer is not None:
                        try:
                            rgb = try_capture(trainer.viewer)
                        except Exception as e:
                            rgb = None
                    # If external x11grab recorder is active, don't call viewer capture to avoid segfaults.
                    if use_external_recorder and not use_pipe:
                        # we intentionally do not collect frames in Python; ffmpeg records the X display.
                        step += 1
                        if args.delay > 0:
                            time.sleep(args.delay)
                        continue
                    if rgb is None and not use_pipe:
                        # try to capture from viewer only if it exists
                        if hasattr(trainer, 'viewer') and trainer.viewer is not None:
                            try:
                                rgb = try_capture(trainer.viewer)
                            except RuntimeError as e:
                                print('Frame capture failed:', e)
                                try:
                                    trainer.close_viewer()
                                except Exception:
                                    pass
                                return
                        else:
                            print('No viewer available to capture frame; skipping and exiting.')
                            return
                    else:
                        # try to capture from viewer only if it exists
                        if hasattr(trainer, 'viewer') and trainer.viewer is not None:
                            try:
                                rgb = try_capture(trainer.viewer)
                            except RuntimeError as e:
                                print('Frame capture failed:', e)
                                try:
                                    trainer.close_viewer()
                                except Exception:
                                    pass
                                return
                        else:
                            print('No viewer available to capture frame; skipping and exiting.')
                            return

                # convert to uint8 if needed
                if use_pipe and rgb is not None:
                    if rgb.dtype != np.uint8:
                        rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8) if rgb.max() <= 1.0 else rgb.astype(np.uint8)
                    # ensure size matches grab size
                    h, w = rgb.shape[:2]
                    expected_h, expected_w = grab_h, grab_w
                    if (w, h) != (expected_w, expected_h):
                        # try to resize using numpy (simple center-crop or pad)
                        # center-crop if larger
                        if w >= expected_w and h >= expected_h:
                            sx = (w - expected_w) // 2
                            sy = (h - expected_h) // 2
                            rgb = rgb[sy:sy+expected_h, sx:sx+expected_w]
                        else:
                            # pad with zeros
                            new = np.zeros((expected_h, expected_w, 3), dtype=np.uint8)
                            new[:h, :w] = rgb
                            rgb = new
                    try:
                        pipe_proc.stdin.write(rgb.tobytes())
                    except Exception:
                        pass
                    step += 1
                else:
                    if rgb is None:
                        # defensive: nothing to append
                        step += 1
                    else:
                        if rgb.dtype != np.uint8:
                            rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8) if rgb.max() <= 1.0 else rgb.astype(np.uint8)
                        if frames_dir:
                            # save per-frame PNG
                            try:
                                imageio.imwrite(os.path.join(frames_dir, f'frame_{frame_idx:06d}.png'), rgb)
                            except Exception:
                                pass
                            frame_idx += 1
                        else:
                            frames.append(rgb)
                        step += 1
                # If using external ffmpeg recorder, pace the loop to the target fps
                if use_external_recorder:
                    frame_time = 1.0 / max(1.0, float(args.fps))
                    elapsed = time.time() - frame_start
                    if elapsed < frame_time:
                        time.sleep(frame_time - elapsed)
                else:
                    if args.delay > 0:
                        time.sleep(args.delay)

    finally:
        # close viewer if it was started
        try:
            if hasattr(trainer, 'viewer') and trainer.viewer is not None:
                trainer.close_viewer()
        except Exception:
            pass
        # stop external ffmpeg recorder if it was started
        try:
            if external_recorder_proc is not None:
                # send 'q' to ask ffmpeg to finish gracefully
                try:
                    external_recorder_proc.communicate(input=b'q', timeout=5)
                except Exception:
                    try:
                        external_recorder_proc.terminate()
                    except Exception:
                        pass
        except Exception:
            pass
        # stop pipe recorder if it was started
        try:
            if pipe_proc is not None:
                try:
                    pipe_proc.stdin.close()
                except Exception:
                    pass
                try:
                    pipe_proc.communicate(timeout=5)
                except Exception:
                    try:
                        pipe_proc.terminate()
                    except Exception:
                        pass
        except Exception:
            pass

    # write video
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fps = args.fps
    if use_external_recorder:
        print(f'External recorder was used; {args.out} should contain the recorded video (ffmpeg wrote file).')
    else:
        if frames_dir:
            # assemble PNGs into video using ffmpeg
            ffmpeg_exe = shutil.which('ffmpeg')
            if ffmpeg_exe is None:
                try:
                    import imageio_ffmpeg as _iioff
                    ffmpeg_exe = _iioff.get_ffmpeg_exe()
                except Exception:
                    ffmpeg_exe = None
            if ffmpeg_exe:
                cmd = [
                    ffmpeg_exe,
                    '-y',
                    '-framerate', str(args.fps),
                    '-i', os.path.join(frames_dir, 'frame_%06d.png'),
                    '-c:v', 'libx264',
                    '-pix_fmt', 'yuv420p',
                    args.out,
                ]
                print('Assembling frames to video with:', ' '.join(cmd))
                try:
                    subprocess.run(cmd, check=True)
                    print('Saved video to', args.out)
                except Exception as e:
                    print('Failed to assemble frames with ffmpeg:', e)
            else:
                print('No ffmpeg found to assemble frames; frames saved in', frames_dir)
        else:
            print(f'Writing {len(frames)} frames to {args.out} at {fps} fps...')
            with imageio.get_writer(args.out, fps=fps) as writer:
                for f in frames:
                    writer.append_data(f)
            print('Saved video.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='results/rollout.mp4', help='output mp4 path')
    parser.add_argument('--steps', type=int, default=800, help='max frames to record')
    parser.add_argument('--max_steps', type=int, default=1000, help='env max steps per episode')
    parser.add_argument('--fps', type=int, default=60)
    parser.add_argument('--delay', type=float, default=0.0, help='sleep seconds between steps (for realtime)')
    parser.add_argument('--width', type=int, default=1400, help='frame width for offscreen render')
    parser.add_argument('--height', type=int, default=800, help='frame height for offscreen render')
    parser.add_argument('--viewer', action='store_true', help='force using MuJoCo viewer for capture (use under xvfb-run)')
    parser.add_argument('--pipe', action='store_true', help='capture frames from viewer and pipe raw frames to ffmpeg stdin (more continuous)')
    parser.add_argument('--frames-dir', type=str, default=None, help='directory to save per-frame PNGs and compose with ffmpeg after run')
    parser.add_argument('--grab-x', type=int, default=0, help='x offset for X11 grab (use to skip left toolbar)')
    parser.add_argument('--grab-y', type=int, default=0, help='y offset for X11 grab')
    parser.add_argument('--grab-width', type=int, default=None, help='width for X11 grab (defaults to --width)')
    parser.add_argument('--grab-height', type=int, default=None, help='height for X11 grab (defaults to --height)')
    args = parser.parse_args()
    main(args)
