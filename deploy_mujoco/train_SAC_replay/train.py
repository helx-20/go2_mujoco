import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

import csv
import random
from datetime import datetime
from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Tuple
import mujoco
import numpy as np
import torch
import yaml
from deploy_mujoco.offline_data_utils import collect_pkl_files, load_chains_from_pkl_file, filter_chain_for_replay
from deploy_mujoco.terrain_trainer import TerrainTrainer

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from model import FullyConnectedQFunction, SamplerPolicy, TanhGaussianPolicy  # noqa: E402
from replay_buffer import ReplayBuffer  # noqa: E402
from sac import SAC  # noqa: E402


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _configure_torch_runtime(cfg: Dict[str, Any]) -> None:
    if bool(cfg.get("torch_deterministic", False)):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = bool(cfg.get("cudnn_benchmark", True))

    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = bool(cfg.get("allow_tf32", True))
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = bool(cfg.get("allow_tf32", True))

    n_threads = int(cfg.get("torch_num_threads", 0))
    if n_threads > 0:
        torch.set_num_threads(n_threads)


def _collect_replay_chains(paths: Sequence[str], keep_k: int) -> List[List[Dict[str, Any]]]:
    files = collect_pkl_files(paths)
    chains_all: List[List[Dict[str, Any]]] = []
    for fp in files:
        try:
            chains = load_chains_from_pkl_file(fp, consecutive_fail_keep_k=0)
            for chain in chains:
                filtered = filter_chain_for_replay(chain, consecutive_fail_keep_k=int(keep_k))
                if len(filtered) > 0:
                    chains_all.append(filtered)
        except Exception:
            continue
    return chains_all


def _infer_bump_region(trainer: TerrainTrainer, terrain_action: np.ndarray) -> Dict[str, Any]:
    dims = trainer.terrain_changer.action_dims if isinstance(trainer.terrain_changer.action_dims, dict) else {}
    if "bump" not in dims or int(dims.get("bump", 0)) < 4:
        return {"valid": False}

    action = np.asarray(terrain_action, dtype=np.float32).reshape(-1)
    if action.shape[0] < 4:
        return {"valid": False}

    cx_norm = float(action[0])
    cy_norm = float(action[1])
    radius = float(action[2])

    cfg = trainer.terrain_changer.terrain_config.get("terrain_action", {})
    min_forward_dist = float(cfg.get("min_forward_dist", 0.0))
    max_forward_dist = float(cfg.get("max_forward_dist", 0.0))
    max_lateral = float(cfg.get("max_lateral", 0.0))
    radius_min = float(cfg.get("radius_min", 0.0))
    radius_max = float(cfg.get("radius_max", 0.0))

    robot_xy = np.asarray(trainer.data.qpos[:2], dtype=np.float64)
    lin_vel = np.asarray(trainer.data.qvel[:2], dtype=np.float64)
    speed = float(np.linalg.norm(lin_vel))
    if speed > 1e-3:
        dir_f = lin_vel / speed
    else:
        qw, qx, qy, qz = trainer.data.qpos[3:7]
        yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        dir_f = np.array([np.cos(yaw), np.sin(yaw)], dtype=np.float64)

    dist = min_forward_dist + (cx_norm + 1.0) / 2.0 * (max_forward_dist - min_forward_dist)
    lat = cy_norm * max_lateral
    perp = np.array([-dir_f[1], dir_f[0]], dtype=np.float64)
    target_xy = robot_xy + dir_f * dist + perp * lat

    gx, gy = trainer.terrain_changer._world_to_grid(float(target_xy[0]), float(target_xy[1]))

    grid_res = float(trainer.terrain_changer.grid_resolution)
    radius_grid_min = radius_min / max(grid_res, 1e-8)
    radius_grid_max = radius_max / max(grid_res, 1e-8)
    radius_grid = (radius + 1.0) / 2.0 * (radius_grid_max - radius_grid_min) + radius_grid_min

    r0 = int(np.clip(np.floor(gx - radius_grid), 0, trainer.terrain_changer.nrow))
    r1 = int(np.clip(np.ceil(gx + radius_grid + 1), 0, trainer.terrain_changer.nrow))
    c0 = int(np.clip(np.floor(gy - radius_grid), 0, trainer.terrain_changer.ncol))
    c1 = int(np.clip(np.ceil(gy + radius_grid + 1), 0, trainer.terrain_changer.ncol))

    if r1 <= r0 or c1 <= c0:
        return {"valid": False}

    return {
        "valid": True,
        "center_world": [float(target_xy[0]), float(target_xy[1])],
        "center_grid": [int(gx), int(gy)],
        "radius_m": float(radius_grid * grid_res),
        "radius_grid": float(radius_grid),
        "bbox": [r0, r1, c0, c1],
    }


class FrameReplayCollector:
    """Collect transitions by replaying frame-start states from offline chains."""

    def __init__(self, trainer: TerrainTrainer, replay_chains: List[List[Dict[str, Any]]], random_chain: bool = True):
        self.trainer = trainer
        self.replay_chains = replay_chains
        self.random_chain = bool(random_chain)
        self._chain_cursor = -1
        self._curr_chain: Optional[List[Dict[str, Any]]] = None
        self._frame_idx = 0

    def set_sample_policy(self, sample_policy):
        self.sample_policy = sample_policy

    def _pick_chain(self) -> List[Dict[str, Any]]:
        if len(self.replay_chains) == 0:
            raise RuntimeError("No replay chains loaded")
        if self.random_chain:
            return random.choice(self.replay_chains)
        self._chain_cursor = (self._chain_cursor + 1) % len(self.replay_chains)
        return self.replay_chains[self._chain_cursor]

    def _restore_robot_state(self, state: Optional[Dict[str, Any]]) -> bool:
        if not isinstance(state, dict):
            return None

        qpos = np.asarray(state.get("qpos", []), dtype=np.float64)
        qvel = np.asarray(state.get("qvel", []), dtype=np.float64)

        if qpos.shape[0] != self.trainer.data.qpos.shape[0] or qvel.shape[0] != self.trainer.data.qvel.shape[0]:
            return None

        self.trainer.data.qpos[:] = qpos
        self.trainer.data.qvel[:] = qvel
        # mujoco.mj_forward(self.trainer.model, self.trainer.data)
        return qpos, qvel

    def _extract_robot_state_from_transition(self, tr: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        info = tr.get("info", {}) if isinstance(tr.get("info"), dict) else {}
        trace = info.get("go2_rollout_trace", {}) if isinstance(info.get("go2_rollout_trace"), dict) else {}
        states = trace.get("states", [])
        if isinstance(states, list) and len(states) > 0 and isinstance(states[0], dict):
            return states[0]

        top = tr.get("robot_state_before_action")
        if isinstance(top, dict):
            return top

        nested = info.get("robot_state_before_action")
        if isinstance(nested, dict):
            return nested

        return None

    def _extract_obs_from_transition(self, tr: Dict[str, Any]) -> np.ndarray:
        obs = np.asarray(tr.get("obs", []), dtype=np.float32)
        if obs.ndim == 1 and obs.shape[0] == self.trainer.get_terrain_observation().shape[0]:
            return obs
        return self.trainer.get_terrain_observation().astype(np.float32)

    def reset_episode(self):
        self.trainer.reset()
        self._curr_chain = self._pick_chain()
        self._frame_idx = 0

        if len(self._curr_chain) == 0:
            obs = self.trainer.get_terrain_observation().astype(np.float32)
            return obs

        first_tr = self._curr_chain[0]
        obs = self._extract_obs_from_transition(first_tr)

        return obs

    def step(self):
        if self._curr_chain is None or len(self._curr_chain) == 0:
            raise RuntimeError("Replay chain is empty; call reset_episode first")

        # 获取_frame_idx时刻的离线数据
        tr = self._curr_chain[self._frame_idx]
        obs = self._extract_obs_from_transition(tr)
        action_in_tr = tr["action"]
        robot_state = self._extract_robot_state_from_transition(tr)

        # 更新机器人状态，与当前离线数据同步
        qpos, qvel = self._restore_robot_state(robot_state)

        # 计算地形动作
        action = self.sample_policy(np.expand_dims(obs, 0), deterministic=False)[0]

        # 根据当前机器人状态更新地形
        restore_info = self.trainer.terrain_changer.apply_action_vector_with_restore(action)
        self.trainer.terrain_changer._refresh_terrain_safe()
        self.trainer.render_hfield()

        # 执行仿真
        next_obs_sim, _, reward, done_sim, info = self.trainer.step_only_robot(self.trainer.terrain_decimation)
        next_obs = np.asarray(next_obs_sim, dtype=np.float32)
        chain_done = bool(tr.get("done", False))
        self._frame_idx += 1
        replay_done = self._frame_idx >= len(self._curr_chain)
        done = bool(done_sim or replay_done or chain_done)

        # 记录info
        info_out = dict(info) if isinstance(info, dict) else {}
        info_out["replay_frame_idx"] = int(self._frame_idx)
        info_out["replay_done"] = bool(replay_done)
        info_out["env_done"] = bool(done_sim)
        info_out["chain_done"] = bool(chain_done)

        # 恢复地形状态至上一帧，并执行离线数据的地形动作
        if not done:
            self.trainer.terrain_changer.set_restore_bump(restore_info)
            self.trainer.terrain_changer.apply_action_vector_with_robot(qpos, qvel, action_in_tr)
            self.trainer.terrain_changer._refresh_terrain()
            self.trainer.render_hfield()

        return next_obs, action, float(reward), done, info_out, replay_done


class CsvLogger:
    def __init__(self, out_csv: str):
        self.out_csv = out_csv
        self.rows: List[Dict[str, Any]] = []

    def add(self, row: Dict[str, Any]) -> None:
        self.rows.append(row)

    def flush(self) -> None:
        if len(self.rows) == 0:
            return
        keys = list(self.rows[0].keys())
        write_header = not os.path.exists(self.out_csv)
        with open(self.out_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            if write_header:
                writer.writeheader()
            for row in self.rows:
                writer.writerow(row)
        self.rows = []


def _save_checkpoint(
    ckpt_path: str,
    sac: SAC,
    replay_buffer: ReplayBuffer,
    step: int,
    cfg: Dict[str, Any],
) -> None:
    payload = {
        "step": int(step),
        "policy": sac.policy.state_dict(),
        "qf1": sac.qf1.state_dict(),
        "qf2": sac.qf2.state_dict(),
        "target_qf1": sac.target_qf1.state_dict(),
        "target_qf2": sac.target_qf2.state_dict(),
        "sac_config": dict(sac.config),
        "train_config": cfg,
        "buffer_size": int(replay_buffer.size),
        "buffer_ptr": int(replay_buffer.ptr),
    }
    torch.save(payload, ckpt_path)


def _infer_done_reason(info: Dict[str, Any]) -> str:
    if bool(info.get("out_of_terrain_edge", False)):
        return "terrain_edge"
    if bool(info.get("fallen", False)):
        return "fall"
    if bool(info.get("base_collision", False)):
        return "base_collision"
    return "other_done"


def run_eval_rollouts(
    trainer: TerrainTrainer,
    sample_policy: SamplerPolicy,
    episodes: int,
    max_episode_steps: int,
    deterministic: bool,
) -> Dict[str, Any]:
    episodes = max(1, int(episodes))
    max_episode_steps = max(1, int(max_episode_steps))

    returns: List[float] = []
    lengths: List[int] = []
    done_reasons: Dict[str, int] = {
        "terrain_edge": 0,
        "fall": 0,
        "base_collision": 0,
        "other_done": 0,
        "timeout": 0,
    }

    fallen_eps = 0
    base_collision_eps = 0
    thigh_collision_eps = 0
    stuck_eps = 0
    speed_values: List[float] = []

    for _ in range(episodes):
        obs = np.asarray(trainer.reset(), dtype=np.float32)
        ep_return = 0.0
        ep_len = 0
        ep_fallen = False
        ep_base_collision = False
        ep_thigh_collision = False
        ep_stuck = False
        terminated = False

        for _t in range(max_episode_steps):
            action = sample_policy(np.expand_dims(obs, 0), deterministic=deterministic)[0]
            next_obs, _, reward, done, info = trainer.step(action)
            obs = np.asarray(next_obs, dtype=np.float32)

            ep_return += float(reward)
            ep_len += 1

            ep_fallen = ep_fallen or bool(info.get("fallen", False))
            ep_base_collision = ep_base_collision or bool(info.get("base_collision", False))
            ep_thigh_collision = ep_thigh_collision or bool(info.get("thigh_collision", False))
            ep_stuck = ep_stuck or bool(info.get("stuck", False))
            if "speed" in info:
                speed_values.append(float(info["speed"]))

            if bool(done):
                done_reasons[_infer_done_reason(info)] += 1
                terminated = True
                break

        if not terminated:
            done_reasons["timeout"] += 1

        returns.append(ep_return)
        lengths.append(ep_len)
        fallen_eps += int(ep_fallen)
        base_collision_eps += int(ep_base_collision)
        thigh_collision_eps += int(ep_thigh_collision)
        stuck_eps += int(ep_stuck)

    returns_np = np.asarray(returns, dtype=np.float64)
    lengths_np = np.asarray(lengths, dtype=np.float64)
    total_steps = max(1.0, float(lengths_np.sum()))

    return {
        "eval_episodes": int(episodes),
        "eval_return_mean": float(np.mean(returns_np)),
        "eval_return_std": float(np.std(returns_np)),
        "eval_return_min": float(np.min(returns_np)),
        "eval_return_max": float(np.max(returns_np)),
        "eval_episode_len_mean": float(np.mean(lengths_np)),
        "eval_reward_per_step": float(float(np.sum(returns_np)) / total_steps),
        "eval_terminated_rate": float(1.0 - (done_reasons["timeout"] / float(episodes))),
        "eval_timeout_rate": float(done_reasons["timeout"] / float(episodes)),
        "eval_fall_rate": float(fallen_eps / float(episodes)),
        "eval_base_collision_rate": float(base_collision_eps / float(episodes)),
        "eval_thigh_collision_rate": float(thigh_collision_eps / float(episodes)),
        "eval_stuck_rate": float(stuck_eps / float(episodes)),
        "eval_speed_mean": float(np.mean(np.asarray(speed_values, dtype=np.float64))) if len(speed_values) > 0 else np.nan,
        "eval_done_terrain_edge_rate": float(done_reasons["terrain_edge"] / float(episodes)),
        "eval_done_fall_rate": float(done_reasons["fall"] / float(episodes)),
        "eval_done_base_collision_rate": float(done_reasons["base_collision"] / float(episodes)),
        "eval_done_other_rate": float(done_reasons["other_done"] / float(episodes)),
    }


def train(cfg: Dict[str, Any]) -> None:
    _configure_torch_runtime(cfg)
    _set_seed(int(cfg.get("seed", 0)))

    run_name = f"{str(cfg.get('log_name', 'sac_replay_default'))}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = os.path.join(CURRENT_DIR, "train_logs", run_name)
    os.makedirs(log_dir, exist_ok=True)
    print(f"[train_SAC_replay] run_dir: {log_dir}")
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_cfg_path = os.path.join(CURRENT_DIR, "train_config.yaml")
    if os.path.exists(train_cfg_path):
        with open(train_cfg_path, "r", encoding="utf-8") as src, open(
            os.path.join(log_dir, "train_config.yaml"), "w", encoding="utf-8"
        ) as dst:
            dst.write(src.read())

    go2_cfg = [str(cfg.get("go2_task", "terrain")), str(cfg.get("go2_config", "go2.yaml"))]
    terrain_cfg_name = str(cfg.get("terrain_config", "terrain_config.yaml"))
    terrain_cfg_local = os.path.join(CURRENT_DIR, terrain_cfg_name)
    if os.path.exists(terrain_cfg_local):
        terrain_cfg_rel = f"train_SAC_replay/{terrain_cfg_name}"
    else:
        # Fallback to train_SAC terrain config for compatibility with old setups.
        terrain_cfg_rel = f"train_SAC/{terrain_cfg_name}"
    trainer = TerrainTrainer(go2_cfg, terrain_cfg_rel)

    eval_enabled = bool(cfg.get("eval_enabled", False))
    eval_every_updates = max(1, int(cfg.get("eval_every_updates", 1)))
    eval_episodes = max(1, int(cfg.get("eval_episodes", 5)))
    eval_max_episode_steps = max(1, int(cfg.get("eval_max_episode_steps", 200)))
    eval_deterministic = bool(cfg.get("eval_deterministic", True))
    eval_trainer = TerrainTrainer(go2_cfg, terrain_cfg_rel) if eval_enabled else None

    replay_paths = [
        p if os.path.isabs(p) else os.path.normpath(os.path.join(CURRENT_DIR, p))
        for p in cfg.get("replay_pkl_paths", [])
    ]
    replay_chains = _collect_replay_chains(replay_paths, keep_k=int(cfg.get("consecutive_fail_keep_k", 0)))
    if len(replay_chains) == 0:
        raise RuntimeError("No replay chains loaded from replay_pkl_paths")

    collector = FrameReplayCollector(
        trainer=trainer,
        replay_chains=replay_chains,
        random_chain=bool(cfg.get("replay_random_chain", True)),
    )

    obs0 = collector.reset_episode()
    obs_dim = int(obs0.shape[0])
    act_dim = int(trainer.total_action_dims)
    if act_dim <= 0:
        raise RuntimeError("Terrain action dimension is 0; check terrain_config terrain_types")

    device = str(cfg.get("device", "cpu"))
    replay_buffer = ReplayBuffer(obs_dim, act_dim, int(cfg.get("buffer_size", 1_000_000)), device=device)

    policy = TanhGaussianPolicy(
        observation_dim=obs_dim,
        action_dim=act_dim,
        arch=str(cfg.get("policy_arch", "256-256")),
        log_std_multiplier=float(cfg.get("policy_log_std_multiplier", 1.0)),
        log_std_offset=float(cfg.get("policy_log_std_offset", -1.0)),
        orthogonal_init=bool(cfg.get("orthogonal_init", False)),
    )
    qf1 = FullyConnectedQFunction(obs_dim, act_dim, arch=str(cfg.get("qf_arch", "256-256")), orthogonal_init=bool(cfg.get("orthogonal_init", False)))
    qf2 = FullyConnectedQFunction(obs_dim, act_dim, arch=str(cfg.get("qf_arch", "256-256")), orthogonal_init=bool(cfg.get("orthogonal_init", False)))
    target_qf1 = deepcopy(qf1)
    target_qf2 = deepcopy(qf2)

    sac_cfg = dict(
        discount=float(cfg.get("gamma", 0.99)),
        reward_scale=float(cfg.get("reward_scale", 1.0)),
        policy_lr=float(cfg.get("learning_rate", 3e-4)),
        qf_lr=float(cfg.get("learning_rate", 3e-4)),
        soft_target_update_rate=float(cfg.get("tau", 0.005)),
        target_update_period=1,
        use_automatic_entropy_tuning=bool(cfg.get("use_automatic_entropy_tuning", True)),
    )
    target_entropy = float(cfg.get("target_entropy", 0.0))
    if target_entropy >= 0.0:
        target_entropy = -float(act_dim)
    sac_cfg["target_entropy"] = target_entropy

    sac = SAC(sac_cfg, policy, qf1, qf2, target_qf1, target_qf2)
    sac.torch_to_device(device)
    sampler_policy = SamplerPolicy(policy, device)
    collector.set_sample_policy(sampler_policy)

    csv_logger = CsvLogger(os.path.join(log_dir, "metrics.csv"))
    eval_csv_logger = CsvLogger(os.path.join(log_dir, "eval_metrics.csv")) if eval_enabled else None
    total_timesteps = int(cfg.get("total_timesteps", 80_000))
    batch_size = int(cfg.get("batch_size", 256))
    learning_starts = int(cfg.get("learning_starts", 1000))
    train_freq = int(cfg.get("train_freq", 1))
    gradient_steps = int(cfg.get("gradient_steps", 1))
    checkpoint_every_steps = int(cfg.get("checkpoint_every_steps", 10000))
    log_every_steps = int(cfg.get("log_every_steps", 200))

    episode_return = 0.0
    episode_len = 0
    episodes = 0
    updates = 0
    obs = obs0

    for step in range(1, total_timesteps + 1):

        next_obs, action, reward, done, info, replay_done = collector.step()

        replay_buffer.append(obs, action, reward, next_obs, float(done))
        episode_return += float(reward)
        episode_len += 1

        train_metrics: Dict[str, Any] = {}
        if replay_buffer.size >= learning_starts and step % train_freq == 0:
            for grad_i in range(gradient_steps):
                batch = replay_buffer.sample(batch_size)
                out = sac.train(batch)
                if grad_i == gradient_steps - 1:
                    train_metrics = out

            updates += 1
            if eval_enabled and eval_trainer is not None and eval_csv_logger is not None and updates % eval_every_updates == 0:
                eval_metrics = run_eval_rollouts(
                    trainer=eval_trainer,
                    sample_policy=sampler_policy,
                    episodes=eval_episodes,
                    max_episode_steps=eval_max_episode_steps,
                    deterministic=eval_deterministic,
                )
                eval_metrics.update(
                    {
                        "step": int(step),
                        "update": int(updates),
                        "buffer_size": int(replay_buffer.size),
                        "deterministic": bool(eval_deterministic),
                    }
                )
                eval_csv_logger.add(eval_metrics)
                eval_csv_logger.flush()
                print(
                    f"[train_SAC_replay][eval] step={step} update={updates} "
                    f"return_mean={eval_metrics['eval_return_mean']:.4f} "
                    f"fall_rate={eval_metrics['eval_fall_rate']:.3f} "
                    f"base_collision_rate={eval_metrics['eval_base_collision_rate']:.3f}"
                )

        # TODO 这样只能比较step average reward了
        if done:
            episodes += 1
            csv_logger.add(
                {
                    "step": int(step),
                    "episode": int(episodes),
                    "episode_return": float(episode_return),
                    "episode_len": int(episode_len),
                    "buffer_size": int(replay_buffer.size),
                    "env_done": bool(info.get("env_done", False)),
                    "chain_done": bool(info.get("chain_done", False)),
                    "replay_done": bool(info.get("replay_done", False)),
                    "policy_loss": float(train_metrics.get("policy_loss", np.nan)),
                    "qf1_loss": float(train_metrics.get("qf1_loss", np.nan)),
                    "qf2_loss": float(train_metrics.get("qf2_loss", np.nan)),
                    "alpha": float(train_metrics.get("alpha", np.nan)),
                }
            )
            obs = collector.reset_episode()
            episode_return = 0.0
            episode_len = 0

        if step % log_every_steps == 0:
            csv_logger.flush()
            print(
                f"[train_SAC_replay] step={step} episodes={episodes} "
                f"buffer={replay_buffer.size} last_reward={reward:.4f}"
            )

        if checkpoint_every_steps > 0 and step % checkpoint_every_steps == 0:
            ckpt = os.path.join(checkpoint_dir, f"checkpoint_step_{step}.pt")
            _save_checkpoint(ckpt, sac, replay_buffer, step, cfg)

    csv_logger.flush()
    if eval_csv_logger is not None:
        eval_csv_logger.flush()
    _save_checkpoint(os.path.join(log_dir, "model_final.pt"), sac, replay_buffer, total_timesteps, cfg)
    trainer.close_viewer()
    if eval_trainer is not None:
        eval_trainer.close_viewer()


def main() -> None:
    cfg_path = os.path.join(CURRENT_DIR, "train_config.yaml")
    cfg = _load_yaml(cfg_path)
    train(cfg)


if __name__ == "__main__":
    main()


