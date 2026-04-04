import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

import pickle
import shutil
from datetime import datetime
import numpy as np
import torch
import yaml
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecCheckNan
from deploy_mujoco.terrain_trainer import TerrainTrainer, TerrainGymEnv
from deploy_mujoco.offline_data_utils import collect_pkl_files, load_chains_from_pkl_file, filter_chain_for_replay
from deploy_mujoco.reward_recompute_utils import load_reward_cfg_from_yaml, recompute_reward_from_info
from deploy_mujoco.train_SAC_dense.callbacks import DenseTrainingLogger
from deploy_mujoco.train_SAC_dense.dense_replay_buffer import FailureReplayBuffer

def _extract_reward_cfg_from_terrain_yaml(terrain_cfg_file: str):
    return load_reward_cfg_from_yaml(terrain_cfg_file)


def _recompute_reward_from_info(info: dict, reward_cfg: dict) -> float:
    return recompute_reward_from_info(info, reward_cfg)


class FailureRecordingWrapper(gym.Wrapper):
    """Record failure episodes during training and dump to PKL."""

    def __init__(self, env, out_dir: str, pkl_name: str = "train_failure_chains.pkl", flush_every_episodes: int = 50):
        super().__init__(env)
        self.out_dir = out_dir
        self.pkl_path = os.path.join(out_dir, pkl_name)
        self.flush_every_episodes = int(max(1, flush_every_episodes))

        self._curr_obs = None
        self._curr_chain = []
        self._episode_idx = 0
        self._failure_episodes = []

        os.makedirs(self.out_dir, exist_ok=True)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._curr_obs = np.asarray(obs, dtype=np.float32)
        self._curr_chain = []
        return obs, info

    def step(self, action):
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        done = bool(terminated or truncated)

        # info_dict = {
        #     "fallen": bool(info.get("fallen", False)),
        #     "collided": bool(info.get("collided", False)),
        #     "base_collision": bool(info.get("base_collision", False)),
        #     "thigh_collision": bool(info.get("thigh_collision", False)),
        #     "stuck": bool(info.get("stuck", False)),
        #     "terrain_reward": float(info.get("terrain_reward", 0.0)),
        # }

        tr = {
            "obs": np.asarray(self._curr_obs, dtype=np.float32).tolist() if self._curr_obs is not None else [],
            "action": np.asarray(action, dtype=np.float32).tolist(),
            "reward": float(reward),
            "next_obs": np.asarray(next_obs, dtype=np.float32).tolist(),
            "done": done,
            "terrain_control": np.asarray(action, dtype=np.float32).tolist(),
            "info": info,
        }
        self._curr_chain.append(tr)
        self._curr_obs = np.asarray(next_obs, dtype=np.float32)

        if done:
            # TODO
            has_failure = bool(
                info["fallen"]
                or info.get("collided", False)
                or info["base_collision"]
                or info["thigh_collision"]
                or info["stuck"]
            )
            if has_failure:
                self._failure_episodes.append({"episode": int(self._episode_idx), "chain": self._curr_chain})
            self._episode_idx += 1

            if self._episode_idx % self.flush_every_episodes == 0:
                self._flush()

        return next_obs, reward, terminated, truncated, info

    def _flush(self):
        with open(self.pkl_path, "wb") as f:
            pickle.dump(self._failure_episodes, f)

    def close(self):
        self._flush()
        return self.env.close()


class FiniteValueWrapper(gym.Wrapper):
    """Clamp non-finite obs/reward/action to keep SAC numerically stable."""

    def __init__(self, env: gym.Env, obs_abs_clip: float = 1e3):
        super().__init__(env)
        self.obs_abs_clip = float(obs_abs_clip)

    def _sanitize_obs(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=self.obs_abs_clip, neginf=-self.obs_abs_clip)
        return np.clip(obs, -self.obs_abs_clip, self.obs_abs_clip)

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._sanitize_obs(obs), info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.size > 0:
            action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
            action = np.clip(action, -1.0, 1.0)
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._sanitize_obs(obs)
        reward = float(np.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0))
        return obs, reward, terminated, truncated, info


def configure_torch_runtime(cfg: dict):
    """Apply torch/cuda runtime knobs from train_config."""
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


def preload_replay_buffer_from_pkl(model, pkl_paths, reward_cfg=None, consecutive_fail_keep_k: int = 0):
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
                    reward = _recompute_reward_from_info(info, reward_cfg or {})
                    if not (reward_cfg or {}):
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

    print(
        f"[train_SAC_dense] replay preload inserted transitions: {inserted}, "
        f"reward_recompute={'on' if (reward_cfg or {}) else 'off'}, fail_keep_k={int(consecutive_fail_keep_k)}"
    )


def train_sac_dense(
    go2_cfg,
    terrain_cfg,
    total_timesteps=20000,
    max_episode_steps=35,
    log_dir="train_terrain_logs",
    reward_threshold=8.0,
    dense_sample_ratio=0.7,
    learning_starts=10000,
    device="auto",
    learning_rate=3e-4,
    batch_size=256,
    buffer_size=1_000_000,
    train_freq=1,
    gradient_steps=1,
    tau=0.005,
    gamma=0.99,
    seed=0,
    obs_abs_clip=1e3,
    plot_save_every_steps=2000,
    plot_smooth_window=20,
    checkpoint_every_steps=10000,
    preload_model_path="",
    preload_pkl_paths=None,
    failure_pkl_name="train_failure_chains.pkl",
    failure_flush_every_episodes=50,
    reward_cfg=None,
    consecutive_fail_keep_k: int = 0,
):
    preload_pkl_paths = preload_pkl_paths or []
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    def make_env():
        trainer = TerrainTrainer(go2_cfg, terrain_cfg)
        env = TerrainGymEnv(trainer, max_episode_steps=max_episode_steps)
        env = FiniteValueWrapper(env, obs_abs_clip=obs_abs_clip)
        env = Monitor(env)
        env = FailureRecordingWrapper(
            env,
            out_dir=log_dir,
            pkl_name=failure_pkl_name,
            flush_every_episodes=failure_flush_every_episodes,
        )
        return env

    vec_env = DummyVecEnv([make_env])
    vec_env = VecCheckNan(vec_env, raise_exception=False, check_inf=True)

    action_dim = int(np.prod(vec_env.action_space.shape))
    if action_dim <= 0:
        raise ValueError(
            "Terrain action dimension is 0, SAC dense cannot train with empty action space. "
            "Please check terrain config and ensure terrain_action.terrain_types includes controllable types (e.g. ['bump'])."
        )

    model = SAC(
        "MlpPolicy",
        vec_env,
        verbose=1,
        device=device,
        learning_rate=float(learning_rate),
        batch_size=int(batch_size),
        buffer_size=int(buffer_size),
        learning_starts=int(learning_starts),
        train_freq=int(train_freq),
        gradient_steps=int(gradient_steps),
        tau=float(tau),
        gamma=float(gamma),
        replay_buffer_class=FailureReplayBuffer,
        replay_buffer_kwargs={
            "reward_threshold": float(reward_threshold),
            "dense_sample_ratio": dense_sample_ratio,
            "consecutive_fail_keep_k": int(consecutive_fail_keep_k),
        },
        seed=int(seed),
    )

    if preload_model_path and os.path.exists(preload_model_path):
        print(f"[train_SAC_dense] warm-loading pretrained weights from: {preload_model_path}")
        model.set_parameters(preload_model_path, exact_match=False, device=device)

    if preload_pkl_paths:
        preload_replay_buffer_from_pkl(
            model,
            preload_pkl_paths,
            reward_cfg=reward_cfg,
            consecutive_fail_keep_k=int(consecutive_fail_keep_k),
        )

    callback = DenseTrainingLogger(
        out_dir=log_dir,
        save_every_steps=int(plot_save_every_steps),
        smooth_window=int(plot_smooth_window),
        checkpoint_every_steps=int(checkpoint_every_steps),
        checkpoint_start_after_steps=int(learning_starts),
        checkpoint_dir=checkpoint_dir,
    )

    model.learn(total_timesteps=total_timesteps,
                callback=callback)

    model.save(os.path.join(log_dir, "model.zip"))


def main():
    current_path = os.path.dirname(os.path.abspath(__file__))
    train_config_file = "train_config.yaml"
    with open(f"{current_path}/{train_config_file}", "r", encoding="utf-8") as f:
        train_config = yaml.load(f, Loader=yaml.FullLoader)

    configure_torch_runtime(train_config)

    run_name = f"{train_config['log_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = f"{current_path}/train_logs/{run_name}"
    os.makedirs(log_dir, exist_ok=True)
    print(f"[train_SAC_dense] run_dir: {log_dir}")
    terrain_cfg_file = os.path.join(current_path, train_config["terrain_config"])
    reward_cfg = _extract_reward_cfg_from_terrain_yaml(terrain_cfg_file)
    train_cfg_file = os.path.join(current_path, train_config_file)
    shutil.copy2(terrain_cfg_file, log_dir)
    shutil.copy2(train_cfg_file, log_dir)

    go2_cfg = [train_config["go2_task"], train_config["go2_config"]]
    terrain_cfg = f"train_SAC_dense/{train_config['terrain_config']}"

    preload_pkl_paths = train_config.get("preload_pkl_paths", [])
    for i in range(len(preload_pkl_paths)):
        preload_pkl_paths[i] = os.path.join(current_path, preload_pkl_paths[i])

    train_sac_dense(
        go2_cfg,
        terrain_cfg,
        total_timesteps=train_config["total_timesteps"],
        max_episode_steps=train_config["max_episode_steps"],
        log_dir=log_dir,
        reward_threshold=train_config.get("reward_threshold", 8.0),
        dense_sample_ratio=train_config.get("dense_sample_ratio", 0.7),
        learning_starts=train_config.get("learning_starts", 10000),
        device=train_config.get("device", "auto"),
        learning_rate=train_config.get("learning_rate", 1e-4),
        batch_size=train_config.get("batch_size", 256),
        buffer_size=train_config.get("buffer_size", 1_000_000),
        train_freq=train_config.get("train_freq", 1),
        gradient_steps=train_config.get("gradient_steps", 1),
        tau=train_config.get("tau", 0.005),
        gamma=train_config.get("gamma", 0.99),
        seed=train_config.get("seed", 0),
        obs_abs_clip=train_config.get("obs_abs_clip", 1e3),
        plot_save_every_steps=train_config.get("plot_save_every_steps", 2000),
        plot_smooth_window=train_config.get("plot_smooth_window", 20),
        checkpoint_every_steps=train_config.get("checkpoint_every_steps", 10000),
        preload_model_path=train_config.get("preload_model_path", ""),
        preload_pkl_paths=preload_pkl_paths,
        failure_pkl_name=train_config.get("failure_pkl_name", "train_failure_chains.pkl"),
        failure_flush_every_episodes=train_config.get("failure_flush_every_episodes", 50),
        reward_cfg=reward_cfg,
        consecutive_fail_keep_k=int(train_config.get("consecutive_fail_keep_k", 0)),
    )


if __name__ == "__main__":
    main()
