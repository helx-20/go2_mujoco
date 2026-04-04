import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

import csv
import shutil
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
import torch
import yaml
from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure as configure_sb3_logger
from stable_baselines3.common.vec_env import DummyVecEnv
from deploy_mujoco.offline_data_utils import collect_pkl_files, load_chains_from_pkl_file, filter_chain_for_replay
from deploy_mujoco.reward_recompute_utils import load_reward_cfg_from_yaml, recompute_reward_from_info
from deploy_mujoco.terrain_trainer import TerrainGymEnv, TerrainTrainer


def configure_torch_runtime(cfg: dict) -> None:
    seed = int(cfg.get("seed", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)

    deterministic = bool(cfg.get("torch_deterministic", False))
    cudnn_benchmark = bool(cfg.get("cudnn_benchmark", True))
    allow_tf32 = bool(cfg.get("allow_tf32", True))

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32

    num_threads = int(cfg.get("torch_num_threads", 0))
    if num_threads > 0:
        torch.set_num_threads(num_threads)


def preload_replay_buffer_from_pkl(
    model: SAC,
    pkl_paths: List[str],
    reward_cfg: Optional[dict] = None,
    use_reward_recompute: bool = False,
    consecutive_fail_keep_k: int = 0,
) -> int:
    files = collect_pkl_files(pkl_paths)
    inserted = 0

    obs_shape = model.observation_space.shape
    act_shape = model.action_space.shape

    for fp in files:
        try:
            chains = load_chains_from_pkl_file(fp, consecutive_fail_keep_k=0)
        except Exception:
            continue

        for chain in chains:
            filtered_chain = filter_chain_for_replay(chain, consecutive_fail_keep_k=int(consecutive_fail_keep_k))
            for tr in filtered_chain:
                try:
                    obs = np.asarray(tr.get("obs", []), dtype=np.float32).reshape(obs_shape)
                    next_obs = np.asarray(tr.get("next_obs", []), dtype=np.float32).reshape(obs_shape)
                    action = np.asarray(tr.get("action", []), dtype=np.float32).reshape(act_shape)
                    info = tr.get("info", {}) if isinstance(tr.get("info", {}), dict) else {}
                    if use_reward_recompute and (reward_cfg or {}):
                        reward = float(recompute_reward_from_info(info, reward_cfg or {}))
                    else:
                        reward = float(tr.get("reward", 0.0))
                    done = float(bool(tr.get("done", False)))

                    model.replay_buffer.add(
                        np.expand_dims(obs, axis=0),
                        np.expand_dims(next_obs, axis=0),
                        np.expand_dims(action, axis=0),
                        np.array([reward], dtype=np.float32),
                        np.array([done], dtype=np.float32),
                        [info],
                    )
                    inserted += 1
                except Exception:
                    continue

    return inserted


class OfflineTrainingLogger:
    def __init__(self, out_dir: str, flush_every_updates: int = 100):
        self.out_dir = out_dir
        self.flush_every_updates = int(max(1, flush_every_updates))
        self.rows: List[Dict[str, float]] = []
        self.csv_path = os.path.join(out_dir, "offline_metrics.csv")
        os.makedirs(out_dir, exist_ok=True)

    def add(self, row: Dict[str, float]) -> None:
        self.rows.append(row)
        if len(self.rows) >= self.flush_every_updates:
            self.flush()

    def flush(self) -> None:
        if len(self.rows) == 0:
            return
        keys = list(self.rows[0].keys())
        write_header = not os.path.exists(self.csv_path)
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            if write_header:
                writer.writeheader()
            for row in self.rows:
                writer.writerow(row)
        self.rows = []


def run_eval_rollouts(
    model: SAC,
    eval_env: TerrainGymEnv,
    episodes: int,
    max_episode_steps: int,
    deterministic: bool,
) -> Dict[str, float]:
    episodes = int(max(1, episodes))
    max_episode_steps = int(max(1, max_episode_steps))

    returns: List[float] = []
    lengths: List[int] = []
    fall_eps = 0
    base_collision_eps = 0
    thigh_collision_eps = 0
    stuck_eps = 0

    for _ in range(episodes):
        obs, _ = eval_env.reset()
        ep_ret = 0.0
        ep_len = 0
        ep_fall = False
        ep_base_collision = False
        ep_thigh_collision = False
        ep_stuck = False

        for _ in range(max_episode_steps):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            ep_ret += float(reward)
            ep_len += 1

            ep_fall = ep_fall or bool(info.get("fallen", False))
            ep_base_collision = ep_base_collision or bool(info.get("base_collision", False))
            ep_thigh_collision = ep_thigh_collision or bool(info.get("thigh_collision", False))
            ep_stuck = ep_stuck or bool(info.get("stuck", False))

            if bool(terminated) or bool(truncated):
                break

        returns.append(ep_ret)
        lengths.append(ep_len)
        fall_eps += int(ep_fall)
        base_collision_eps += int(ep_base_collision)
        thigh_collision_eps += int(ep_thigh_collision)
        stuck_eps += int(ep_stuck)

    returns_np = np.asarray(returns, dtype=np.float64)
    lengths_np = np.asarray(lengths, dtype=np.float64)

    return {
        "eval_episodes": float(episodes),
        "eval_return_mean": float(np.mean(returns_np)),
        "eval_return_std": float(np.std(returns_np)),
        "eval_return_min": float(np.min(returns_np)),
        "eval_return_max": float(np.max(returns_np)),
        "eval_episode_len_mean": float(np.mean(lengths_np)),
        "eval_fall_rate": float(fall_eps / episodes),
        "eval_base_collision_rate": float(base_collision_eps / episodes),
        "eval_thigh_collision_rate": float(thigh_collision_eps / episodes),
        "eval_stuck_rate": float(stuck_eps / episodes),
    }


def train_sac_offline(cfg: dict) -> None:
    configure_torch_runtime(cfg)

    current_path = os.path.dirname(os.path.realpath(__file__))
    run_name = f"{str(cfg.get('log_name', 'sac_offline'))}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = os.path.join(current_path, "train_logs", run_name)
    os.makedirs(log_dir, exist_ok=True)
    print(f"[train_SAC_offline] run_dir: {log_dir}")
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    cfg_file = os.path.join(current_path, "train_config.yaml")
    terrain_cfg_file = os.path.join(current_path, str(cfg.get("terrain_config", "terrain_config.yaml")))
    shutil.copy2(cfg_file, os.path.join(log_dir, "train_config.yaml"))
    if os.path.exists(terrain_cfg_file):
        shutil.copy2(terrain_cfg_file, os.path.join(log_dir, "terrain_config.yaml"))

    go2_cfg = [str(cfg.get("go2_task", "terrain")), str(cfg.get("go2_config", "go2.yaml"))]
    terrain_cfg = f"train_SAC_offline/{str(cfg.get('terrain_config', 'terrain_config.yaml'))}"
    max_episode_steps = int(cfg.get("max_episode_steps", 35))
    eval_enabled = bool(cfg.get("eval_enabled", False))
    eval_every_updates = int(max(1, cfg.get("eval_every_updates", 500)))
    eval_episodes = int(max(1, cfg.get("eval_episodes", 5)))
    eval_max_episode_steps = int(max(1, cfg.get("eval_max_episode_steps", max_episode_steps)))
    eval_deterministic = bool(cfg.get("eval_deterministic", True))
    eval_render = bool(cfg.get("eval_render", False))

    def make_env():
        trainer = TerrainTrainer(go2_cfg, terrain_cfg)
        return TerrainGymEnv(trainer, max_episode_steps=max_episode_steps)

    vec_env = DummyVecEnv([make_env])

    eval_trainer = None
    eval_env = None
    if eval_enabled:
        eval_trainer = TerrainTrainer(go2_cfg, terrain_cfg)
        if not eval_trainer.render and eval_render:
            eval_trainer.render = True
            eval_trainer.start_viewer()
        eval_env = TerrainGymEnv(eval_trainer, max_episode_steps=eval_max_episode_steps)

    action_dim = int(np.prod(vec_env.action_space.shape))
    if action_dim <= 0:
        raise ValueError("Terrain action dimension is 0. Please check terrain_action.terrain_types in terrain_config.")

    model = SAC(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        device=cfg.get("device", "auto"),
        learning_rate=float(cfg.get("learning_rate", 3e-4)),
        batch_size=int(cfg.get("batch_size", 256)),
        buffer_size=int(cfg.get("buffer_size", 1_000_000)),
        learning_starts=0,
        train_freq=1,
        gradient_steps=1,
        tau=float(cfg.get("tau", 0.005)),
        gamma=float(cfg.get("gamma", 0.99)),
        seed=int(cfg.get("seed", 0)),
    )

    # model.learn() would set logger internally; offline loop calls model.train() directly.
    sb3_log_dir = os.path.join(log_dir, "sb3_logs")
    os.makedirs(sb3_log_dir, exist_ok=True)
    model.set_logger(configure_sb3_logger(sb3_log_dir, ["stdout", "csv"]))

    preload_model_path = str(cfg.get("preload_model_path", "")).strip()
    if preload_model_path:
        if not os.path.isabs(preload_model_path):
            preload_model_path = os.path.normpath(os.path.join(current_path, preload_model_path))
        if os.path.exists(preload_model_path):
            print(f"[train_SAC_offline] warm-loading pretrained weights from: {preload_model_path}")
            model.set_parameters(preload_model_path, exact_match=False, device=cfg.get("device", "auto"))

    offline_pkl_paths = [
        p if os.path.isabs(p) else os.path.normpath(os.path.join(current_path, p))
        for p in cfg.get("offline_pkl_paths", [])
    ]

    use_reward_recompute = bool(cfg.get("reward_recompute", False))
    reward_cfg = load_reward_cfg_from_yaml(terrain_cfg_file) if use_reward_recompute else None

    inserted = preload_replay_buffer_from_pkl(
        model,
        offline_pkl_paths,
        reward_cfg=reward_cfg,
        use_reward_recompute=use_reward_recompute,
        consecutive_fail_keep_k=int(cfg.get("consecutive_fail_keep_k", 0)),
    )

    if inserted == 0:
        raise RuntimeError("No transitions inserted from offline_pkl_paths. Check data path and file format.")

    batch_size = int(cfg.get("batch_size", 256))
    gradient_steps_per_update = int(max(1, cfg.get("gradient_steps_per_update", 1)))
    total_updates = int(cfg.get("total_updates", 50_000))
    log_every_updates = int(max(1, cfg.get("log_every_updates", 100)))
    checkpoint_every_updates = int(max(1, cfg.get("checkpoint_every_updates", 2000)))

    if model.replay_buffer.size() < batch_size:
        raise RuntimeError(
            f"Offline buffer size {model.replay_buffer.size()} < batch_size {batch_size}, cannot start training."
        )

    logger = OfflineTrainingLogger(log_dir, flush_every_updates=log_every_updates)
    eval_logger = OfflineTrainingLogger(log_dir, flush_every_updates=1) if eval_enabled else None
    if eval_logger is not None:
        eval_logger.csv_path = os.path.join(log_dir, "eval_metrics.csv")

    print(
        f"[train_SAC_offline] start offline training: updates={total_updates}, "
        f"grad_steps_per_update={gradient_steps_per_update}, replay_size={model.replay_buffer.size()}, "
        f"reward_recompute={'on' if use_reward_recompute else 'off'}, "
        f"fail_keep_k={int(cfg.get('consecutive_fail_keep_k', 0))}"
    )

    for update in range(1, total_updates + 1):
        model.train(gradient_steps=gradient_steps_per_update, batch_size=batch_size)
        log_dict = model.logger.name_to_value if hasattr(model, "logger") else {}

        row = {
            "update": int(update),
            "replay_size": int(model.replay_buffer.size()),
            "actor_loss": float(log_dict.get("train/actor_loss", np.nan)),
            "critic_loss": float(log_dict.get("train/critic_loss", np.nan)),
            "ent_coef": float(log_dict.get("train/ent_coef", np.nan)),
            "ent_coef_loss": float(log_dict.get("train/ent_coef_loss", np.nan)),
        }
        logger.add(row)

        if update % log_every_updates == 0:
            logger.flush()
            print(
                f"[train_SAC_offline] update={update} actor_loss={row['actor_loss']:.6f} "
                f"critic_loss={row['critic_loss']:.6f} ent_coef={row['ent_coef']:.6f}"
            )

        if update % checkpoint_every_updates == 0:
            model.save(os.path.join(checkpoint_dir, f"checkpoint_update_{update}.zip"))

        if eval_enabled and eval_env is not None and eval_logger is not None and update % eval_every_updates == 0:
            eval_row = run_eval_rollouts(
                model=model,
                eval_env=eval_env,
                episodes=eval_episodes,
                max_episode_steps=eval_max_episode_steps,
                deterministic=eval_deterministic,
            )
            eval_row["update"] = float(update)
            eval_logger.add(eval_row)
            eval_logger.flush()
            print(
                f"[train_SAC_offline][eval] update={update} return_mean={eval_row['eval_return_mean']:.6f} "
                f"fall_rate={eval_row['eval_fall_rate']:.3f} base_collision_rate={eval_row['eval_base_collision_rate']:.3f}"
            )

    logger.flush()
    model.save(os.path.join(log_dir, "model.zip"))
    vec_env.close()
    if eval_trainer is not None:
        eval_trainer.close_viewer()


def main() -> None:
    current_path = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(current_path, "train_config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    train_sac_offline(cfg)


if __name__ == "__main__":
    main()

