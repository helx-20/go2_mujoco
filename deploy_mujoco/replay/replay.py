import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

import glob
import pickle
import shutil
import time
from typing import Dict, List, Optional, Sequence, Tuple

import mujoco
import numpy as np
import yaml

from deploy_mujoco.offline_data_utils import collect_pkl_files
from deploy_mujoco.terrain_trainer import TerrainTrainer


def _load_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_pkl(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def _extract_chains(obj) -> List[List[Dict]]:
    chains: List[List[Dict]] = []
    if isinstance(obj, dict):
        if isinstance(obj.get("chain"), list):
            chains.append(obj["chain"])
        elif isinstance(obj.get("chains"), list):
            for c in obj["chains"]:
                if isinstance(c, list):
                    chains.append(c)
    elif isinstance(obj, list):
        if len(obj) == 0:
            return chains
        if isinstance(obj[0], dict) and "chain" in obj[0]:
            for ep in obj:
                c = ep.get("chain", [])
                if isinstance(c, list):
                    chains.append(c)
        elif isinstance(obj[0], dict) and "obs" in obj[0] and "action" in obj[0]:
            chains.append(obj)
    return chains


def _find_yaml_in_dir(log_dir: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Find (terrain_cfg, go2_cfg, train_cfg) from pkl sibling directory."""
    ymls = glob.glob(os.path.join(log_dir, "*.yaml")) + glob.glob(os.path.join(log_dir, "*.yml"))
    terrain_cfg = None
    go2_cfg = None
    train_cfg = None

    for yp in ymls:
        try:
            cfg = _load_yaml(yp)
        except Exception:
            continue
        if not isinstance(cfg, dict):
            continue

        if terrain_cfg is None and isinstance(cfg.get("terrain_action"), dict):
            terrain_cfg = yp

        if go2_cfg is None and "xml_path" in cfg and "simulation_dt" in cfg and "control_decimation" in cfg:
            go2_cfg = yp

        if train_cfg is None and "go2_task" in cfg and "go2_config" in cfg:
            train_cfg = yp

    return terrain_cfg, go2_cfg, train_cfg


def _stage_configs(
    base_dir: str,
    pkl_file: str,
    go2_task: str,
    terrain_cfg_path: str,
    go2_cfg_path: str,
) -> Tuple[List[str], str, str]:
    """Stage configs into expected trainer paths.

    TerrainTrainer expects:
      - go2 config path: <base>/<go2_task>/configs/<go2_file>
      - terrain config path: relative under <base>
    """
    tag = os.path.splitext(os.path.basename(pkl_file))[0]

    # Copy go2 config into task config dir.
    task_cfg_dir = os.path.join(base_dir, go2_task, "configs")
    os.makedirs(task_cfg_dir, exist_ok=True)
    go2_name = f"replay_{tag}_go2.yaml"
    go2_dst = os.path.join(task_cfg_dir, go2_name)
    shutil.copy2(go2_cfg_path, go2_dst)

    # Copy terrain config into replay runtime dir.
    runtime_dir = os.path.join(base_dir, "replay", "_runtime")
    os.makedirs(runtime_dir, exist_ok=True)
    terrain_name = f"replay_{tag}_terrain_config.yaml"
    terrain_dst = os.path.join(runtime_dir, terrain_name)
    shutil.copy2(terrain_cfg_path, terrain_dst)

    go2_tuple = [go2_task, go2_name]
    terrain_rel = f"replay/_runtime/{terrain_name}"
    return go2_tuple, terrain_rel, go2_dst


def _print_config_summary(terrain_cfg: Dict, go2_cfg: Dict) -> None:
    ta = terrain_cfg.get("terrain_action", {}) if isinstance(terrain_cfg, dict) else {}
    types = ta.get("terrain_types", [])
    dt = float(go2_cfg.get("simulation_dt", 0.0))
    ctrl_dec = int(go2_cfg.get("control_decimation", 1))
    terr_dec = int(ta.get("terrain_decimation", 1))
    terr_step_dt = dt * ctrl_dec * terr_dec

    print("[replay] terrain_types:", types)
    if "bump" in types:
        print(
            "[replay] bump: "
            f"forward=[{ta.get('min_forward_dist', 'NA')},{ta.get('max_forward_dist', 'NA')}], "
            f"max_lateral={ta.get('max_lateral', 'NA')}, "
            f"radius=[{ta.get('radius_min', 'NA')},{ta.get('radius_max', 'NA')}], "
            f"max_h={ta.get('max_bump_height', 'NA')}"
        )

    print(
        "[replay] timing: "
        f"simulation_dt={dt}, control_decimation={ctrl_dec}, terrain_decimation={terr_dec}, "
        f"terrain_step_dt={terr_step_dt}"
    )


def _set_state_from_trace(trainer: TerrainTrainer, state_dict: Dict) -> bool:
    qpos = np.asarray(state_dict.get("qpos", []), dtype=np.float64)
    qvel = np.asarray(state_dict.get("qvel", []), dtype=np.float64)
    if qpos.shape[0] != trainer.data.qpos.shape[0]:
        return False
    if qvel.shape[0] != trainer.data.qvel.shape[0]:
        return False

    trainer.data.qpos[:] = qpos
    trainer.data.qvel[:] = qvel
    mujoco.mj_forward(trainer.model, trainer.data)
    if trainer.render and trainer.lock_camera:
        trainer.viewer.cam.lookat[:] = trainer.data.qpos[:3]
    if trainer.render:
        trainer.viewer.sync()
    return True


def _replay_robot_action(
    trainer: TerrainTrainer,
    tr: Dict,
    trace_sleep_dt: float,
) -> bool:
    info = tr.get("info", {}) if isinstance(tr.get("info"), dict) else {}
    trace = info.get("go2_rollout_trace", {}) if isinstance(info.get("go2_rollout_trace"), dict) else {}
    states = trace.get("states", [])
    actions = trace.get("actions", [])
    if len(states) == 0 or len(actions) == 0:
        return False

    if not _set_state_from_trace(trainer, states[0]):
        return False

    for a in actions:
        tau = np.asarray(a.get("tau", []), dtype=np.float64)
        if tau.shape[0] != trainer.data.ctrl.shape[0]:
            return False
        trainer.data.ctrl[:] = tau
        mujoco.mj_step(trainer.model, trainer.data)
        if trainer.render and trainer.lock_camera:
            trainer.viewer.cam.lookat[:] = trainer.data.qpos[:3]
        if trainer.render:
            trainer.viewer.sync()
        if trace_sleep_dt > 0.0:
            time.sleep(trace_sleep_dt)
    return True


def _replay_robot_state(
    trainer: TerrainTrainer,
    tr: Dict,
    trace_sleep_dt: float,
) -> bool:
    info = tr.get("info", {}) if isinstance(tr.get("info"), dict) else {}
    trace = info.get("go2_rollout_trace", {}) if isinstance(info.get("go2_rollout_trace"), dict) else {}
    states = trace.get("states", [])
    if len(states) == 0:
        return False

    for s in states:
        if not _set_state_from_trace(trainer, s):
            return False
        if trace_sleep_dt > 0.0:
            time.sleep(trace_sleep_dt)
    return True


def _run_terrain_action(trainer: TerrainTrainer, tr: Dict) -> None:
    terrain_action = tr.get("terrain_action", tr.get("action", []))
    terrain_action = np.asarray(terrain_action, dtype=np.float32)
    trainer.step(terrain_action)


def _run_robot_only(trainer: TerrainTrainer) -> None:
    # Keep preset terrain fixed while replaying robot dynamics.
    for _ in range(trainer.terrain_decimation):
        trainer.step_only_robot()


def _extract_terrain_action(tr: Dict) -> np.ndarray:
    return np.asarray(tr.get("terrain_action", tr.get("action", [])), dtype=np.float32)


def _apply_terrain_only(trainer: TerrainTrainer, tr: Dict) -> None:
    """Apply terrain action from pkl without advancing robot controller rollout."""
    terrain_action = _extract_terrain_action(tr)

    trainer.terrain_changer.apply_action_vector(terrain_action)
    trainer.terrain_changer._refresh_terrain_safe()

    if trainer.render:
        trainer.viewer.update_hfield(trainer.terrain_changer.hfield_id)
        if trainer.lock_camera:
            trainer.viewer.cam.lookat[:] = trainer.data.qpos[:3]
        trainer.viewer.sync()


def _set_hfield_and_refresh(trainer: TerrainTrainer, hfield_2d: np.ndarray) -> None:
    trainer.terrain_changer.hfield[:, :] = hfield_2d
    trainer.terrain_changer._refresh_terrain_safe()
    if trainer.render:
        trainer.viewer.update_hfield(trainer.terrain_changer.hfield_id)
        if trainer.lock_camera:
            trainer.viewer.cam.lookat[:] = trainer.data.qpos[:3]
        trainer.viewer.sync()


def _compute_preset_hfield(
    trainer: TerrainTrainer,
    chain: List[Dict],
) -> np.ndarray:
    """Build one preset terrain for current pkl by applying all its terrain actions in advance."""
    base = trainer.terrain_changer.original_hfield.reshape(trainer.terrain_changer.nrow, trainer.terrain_changer.ncol)
    trainer.terrain_changer.hfield[:, :] = base

    for tr in chain:
        # Keep terrain-action context aligned with recorded robot state when trace exists.
        info = tr.get("info", {}) if isinstance(tr.get("info"), dict) else {}
        trace = info.get("go2_rollout_trace", {}) if isinstance(info.get("go2_rollout_trace"), dict) else {}
        states = trace.get("states", [])
        qpos = np.asarray(states[0].get("qpos", []), dtype=np.float64)
        qvel = np.asarray(states[0].get("qvel", []), dtype=np.float64)
        trainer.terrain_changer.apply_action_vector_with_robot(qpos, qvel, _extract_terrain_action(tr))

    return np.asarray(trainer.terrain_changer.hfield, dtype=np.float32).copy()


def _replay_one_transition(
    trainer: TerrainTrainer,
    tr: Dict,
    replay_mode: str,
    fallback_to_terrain_action: bool,
    trace_sleep_dt: float,
    terrain_mode: str,
) -> None:
    # terrain_action mode: robot is advanced by trainer API
    if replay_mode == "terrain_action":
        if terrain_mode == "preset":
            _run_robot_only(trainer)
            # _run_terrain_action(trainer, tr)
        else:
            _run_terrain_action(trainer, tr)
        return

    # robot_action / robot_state modes
    if terrain_mode == "realtime":
        _apply_terrain_only(trainer, tr)

    if replay_mode == "robot_action":
        ok = _replay_robot_action(trainer, tr, trace_sleep_dt)
        if ok:
            return
        if not fallback_to_terrain_action:
            return
        if terrain_mode == "preset":
            _run_robot_only(trainer)
            return

    if replay_mode == "robot_state":
        ok = _replay_robot_state(trainer, tr, trace_sleep_dt)
        if ok:
            return
        if not fallback_to_terrain_action:
            return
        if terrain_mode == "preset":
            _run_robot_only(trainer)
            return

    _run_terrain_action(trainer, tr)


def _resolve_go2_task(default_task: str, train_cfg_path: Optional[str]) -> str:
    if train_cfg_path and os.path.exists(train_cfg_path):
        try:
            cfg = _load_yaml(train_cfg_path)
            task = str(cfg.get("go2_task", default_task)).strip()
            if task in ("terrain", "velocity"):
                return task
        except Exception:
            pass
    if default_task in ("terrain", "velocity"):
        return default_task
    return "terrain"


def main() -> None:
    base_dir = os.path.dirname(os.path.realpath(__file__))
    cfg = _load_yaml(os.path.join(base_dir, "replay_config.yaml"))

    preload_pkl_paths = cfg.get("pkl_paths", [])
    for i in range(len(preload_pkl_paths)):
        preload_pkl_paths[i] = os.path.join(base_dir, preload_pkl_paths[i])
    pkl_files = collect_pkl_files(preload_pkl_paths)
    max_files = int(cfg.get("max_files", 0))
    if max_files > 0:
        pkl_files = pkl_files[:max_files]

    if len(pkl_files) == 0:
        raise RuntimeError("No pkl files found from replay_config.yaml:pkl_paths")

    replay_mode = str(cfg.get("replay_mode", "terrain_action")).strip()
    if replay_mode not in ("terrain_action", "robot_action", "robot_state"):
        raise ValueError(f"Unsupported replay_mode: {replay_mode}")

    render = bool(cfg.get("render", True))
    realtime_sim = bool(cfg.get("realtime_sim", False))
    max_eps = int(cfg.get("max_episodes_per_file", 0))
    max_steps = int(cfg.get("max_steps_per_episode", 0))
    fallback_to_terrain_action = bool(cfg.get("fallback_to_terrain_action", True))
    trace_sleep_dt = float(cfg.get("trace_sleep_dt", 0.0))
    default_go2_task = str(cfg.get("go2_task", "terrain"))

    terrain_replay_cfg = cfg.get("terrain_replay", {}) if isinstance(cfg.get("terrain_replay", {}), dict) else {}
    terrain_mode = str(terrain_replay_cfg.get("mode", "realtime")).strip()
    if terrain_mode not in ("realtime", "preset"):
        raise ValueError(f"Unsupported terrain_replay.mode: {terrain_mode}")

    for fp in pkl_files:
        log_dir = os.path.dirname(fp)
        terrain_cfg_path, go2_cfg_path, train_cfg_path = _find_yaml_in_dir(log_dir)
        if terrain_cfg_path is None or go2_cfg_path is None:
            print(f"[replay] skip {fp}: cannot find sibling terrain/go2 yaml")
            continue

        go2_task = _resolve_go2_task(default_go2_task, train_cfg_path)
        go2_tuple, terrain_rel, _ = _stage_configs(
            os.path.dirname(base_dir),
            fp,
            go2_task,
            terrain_cfg_path,
            go2_cfg_path,
        )

        terrain_cfg = _load_yaml(terrain_cfg_path)
        go2_cfg = _load_yaml(go2_cfg_path)

        print(f"\n[replay] file: {fp}")
        print(f"[replay] mode: {replay_mode}, go2_task: {go2_task}, terrain_mode: {terrain_mode}")
        _print_config_summary(terrain_cfg, go2_cfg)

        trainer = TerrainTrainer(go2_tuple, terrain_rel)
        trainer.render = render
        trainer.realtime_sim = realtime_sim
        if render:
            trainer.start_viewer()

        obj = _load_pkl(fp)
        chains = _extract_chains(obj)
        if max_eps > 0:
            chains = chains[:max_eps]

        for ep_idx, chain in enumerate(chains):
            if max_steps > 0:
                chain = chain[:max_steps]

            if terrain_mode == "preset":
                preset_hfield_file = _compute_preset_hfield(
                    trainer,
                    chain,
                )
                trainer.reset()
                _set_hfield_and_refresh(trainer, preset_hfield_file)
            else:
                trainer.reset()

            for tr in chain:
                _replay_one_transition(
                    trainer,
                    tr,
                    replay_mode,
                    fallback_to_terrain_action,
                    trace_sleep_dt,
                    terrain_mode,
                )

        trainer.close_viewer()


if __name__ == "__main__":
    main()

