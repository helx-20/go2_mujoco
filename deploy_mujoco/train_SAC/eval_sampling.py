import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

import csv
from datetime import datetime
import pickle
import shutil
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
import yaml
from stable_baselines3 import SAC
from deploy_mujoco.terrain_trainer import TerrainGymEnv, TerrainTrainer


def _build_log_paths(log_dir: str) -> Dict[str, str]:
    os.makedirs(log_dir, exist_ok=True)
    return {
        "log_dir": log_dir,
        "pkl": os.path.join(log_dir, "collision_failures.pkl"),
        "non_failure_pkl": os.path.join(log_dir, "non_failure_trajectories.pkl"),
        "csv": os.path.join(log_dir, "failure_summary.csv"),
    }


def _append_csv_row(csv_path: str, row: Dict[str, Any]) -> None:
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "episodes_evaluated",
                "total_failures",
                "collision_failures",
                "fall_failures",
                "base_collision_failures",
                "thigh_collision_failures",
                "stuck_failures",
            ],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _sample_uniform_action_with_density(action_space, rng: np.random.Generator) -> Tuple[np.ndarray, float]:
    low = np.asarray(action_space.low, dtype=np.float64)
    high = np.asarray(action_space.high, dtype=np.float64)
    action = rng.uniform(low, high).astype(np.float32)
    width = np.maximum(high - low, 1e-12)
    density = float(1.0 / np.prod(width))
    return action, density


def _sample_sac_action_with_density(model: SAC, obs: np.ndarray) -> Tuple[np.ndarray, float, float]:
    obs_np = np.asarray(obs, dtype=np.float32)
    obs_tensor = torch.as_tensor(obs_np, device=model.device).unsqueeze(0)
    with torch.no_grad():
        action_tensor, log_prob_tensor = model.policy.actor.action_log_prob(obs_tensor)
    action = action_tensor.detach().cpu().numpy().reshape(-1).astype(np.float32)
    log_prob = float(log_prob_tensor.detach().cpu().numpy().reshape(-1)[0])
    density = float(np.exp(np.clip(log_prob, -60.0, 60.0)))
    return action, log_prob, density


def _clip_uniform_weight(weight: float) -> float:
    return float(np.clip(weight, 0.0, 1.0))


def _sample_weighted_mixture_action(
    model: SAC,
    obs: np.ndarray,
    action_space,
    rng: np.random.Generator,
    uniform_weight: float,
) -> Dict[str, Any]:
    uniform_weight = _clip_uniform_weight(uniform_weight)
    sac_action, sac_log_prob, sac_density = _sample_sac_action_with_density(model, obs)
    uniform_action, uniform_density = _sample_uniform_action_with_density(action_space, rng)

    use_uniform = bool(rng.random() < uniform_weight)
    action = uniform_action if use_uniform else sac_action

    # Mixture density p(a) = w * p_u(a) + (1 - w) * p_sac(a), tracked for later analysis.
    mixture_density = float(uniform_weight * uniform_density + (1.0 - uniform_weight) * sac_density)

    return {
        "action": action,
        "action_source": "uniform" if use_uniform else "sac",
        "action_log_prob": None if use_uniform else float(sac_log_prob),
        "action_prob_density": mixture_density,
        "uniform_density": float(uniform_density),
        "sac_density": float(sac_density),
        "uniform_weight": float(uniform_weight),
    }


def _copy_config_snapshots(log_dir: str, go2_cfg_file: str, terrain_cfg_file: str) -> None:
    os.makedirs(log_dir, exist_ok=True)
    go2_cfg_file = str(go2_cfg_file)
    terrain_cfg_file = str(terrain_cfg_file)
    if os.path.exists(go2_cfg_file):
        shutil.copy2(go2_cfg_file, log_dir)
    if os.path.exists(terrain_cfg_file):
        shutil.copy2(terrain_cfg_file, log_dir)


def evaluate_policy_sampling(
    model: Optional[SAC],
    go2_cfg,
    terrain_cfg: str,
    episodes: int,
    max_episode_steps: int,
    log_dir: str,
    seed: int,
    render: bool,
    uniform_weight: float,
    save_non_failure_trajectories: bool = False,
) -> None:
    current_path = os.path.dirname(os.path.realpath(__file__))

    trainer = TerrainTrainer(go2_cfg, terrain_cfg)
    if not trainer.render and render:
        trainer.render = True
        trainer.start_viewer()

    env = TerrainGymEnv(trainer, max_episode_steps=max_episode_steps)
    rng = np.random.default_rng(int(seed))

    paths = _build_log_paths(log_dir)
    go2_cfg_file = os.path.join(current_path, "..", go2_cfg[0], "configs", go2_cfg[1])
    terrain_cfg_file = os.path.join(current_path, "..", terrain_cfg)
    _copy_config_snapshots(paths["log_dir"], go2_cfg_file, terrain_cfg_file)

    summary = {
        "episodes_evaluated": 0,
        "total_failures": 0,
        "collision_failures": 0,
        "fall_failures": 0,
        "base_collision_failures": 0,
        "thigh_collision_failures": 0,
        "stuck_failures": 0,
    }

    failures = []
    non_failures = []

    for ep in range(int(episodes)):
        obs, _ = env.reset()
        ep_chain = []
        has_collision = False
        has_fall = False
        has_base_collision = False
        has_thigh_collision = False
        has_stuck = False

        for step_idx in range(int(max_episode_steps)):
            print(f"Episode {ep + 1}/{episodes}, Step {step_idx + 1}/{max_episode_steps}", end="\r")

            if model is None:
                action, uniform_density = _sample_uniform_action_with_density(env.action_space, rng)
                action_meta = {
                    "action": action,
                    "action_source": "uniform",
                    "action_log_prob": None,
                    "action_prob_density": float(uniform_density),
                    "uniform_density": float(uniform_density),
                    "sac_density": None,
                    "uniform_weight": 1.0,
                }
            else:
                action_meta = _sample_weighted_mixture_action(
                    model=model,
                    obs=obs,
                    action_space=env.action_space,
                    rng=rng,
                    uniform_weight=uniform_weight,
                )

            action = np.asarray(action_meta["action"], dtype=np.float32)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            transition = {
                "obs": np.asarray(obs, dtype=np.float32).tolist(),
                "action": action.tolist(),
                "action_source": action_meta["action_source"],
                "action_log_prob": action_meta["action_log_prob"],
                "action_prob_density": float(action_meta["action_prob_density"]),
                "action_uniform_density": action_meta.get("uniform_density"),
                "action_sac_density": action_meta.get("sac_density"),
                "action_uniform_weight": float(action_meta.get("uniform_weight", 1.0)),
                "reward": float(reward),
                "next_obs": np.asarray(next_obs, dtype=np.float32).tolist(),
                "done": done,
                "terrain_control": action.tolist(),
                "info": info,
            }
            ep_chain.append(transition)

            has_collision = has_collision or bool(info.get("collided", False))
            has_fall = has_fall or bool(info.get("fallen", False))
            has_base_collision = has_base_collision or bool(info.get("base_collision", False))
            has_thigh_collision = has_thigh_collision or bool(info.get("thigh_collision", False))
            has_stuck = has_stuck or bool(info.get("stuck", False))

            obs = next_obs
            if done:
                break

        has_failure = has_collision or has_fall or has_stuck or has_base_collision or has_thigh_collision
        summary["episodes_evaluated"] += 1
        summary["total_failures"] += int(has_failure)
        summary["collision_failures"] += int(has_collision)
        summary["fall_failures"] += int(has_fall)
        summary["base_collision_failures"] += int(has_base_collision)
        summary["thigh_collision_failures"] += int(has_thigh_collision)
        summary["stuck_failures"] += int(has_stuck)

        if has_failure:
            failures.append({"episode": ep, "chain": ep_chain})
            with open(paths["pkl"], "wb") as pf:
                pickle.dump(failures, pf)
        elif save_non_failure_trajectories:
            non_failures.append({"episode": ep, "chain": ep_chain})
            with open(paths["non_failure_pkl"], "wb") as pf:
                pickle.dump(non_failures, pf)

        if summary["episodes_evaluated"] % 100 == 0:
            _append_csv_row(paths["csv"], summary.copy())

    _append_csv_row(paths["csv"], summary.copy())
    trainer.close_viewer()


def _load_eval_context(eval_config: Dict[str, Any], current_path: str):
    policy = str(eval_config["policy"])
    checkpoint = str(eval_config["checkpoint"])

    if policy == "random":
        model = None
    else:
        model_path = os.path.join(current_path, "train_logs", policy, checkpoint)
        model = SAC.load(model_path)

    if policy == "random" or bool(eval_config.get("use_curr_force", False)):
        go2_cfg = [eval_config["go2_task"], eval_config["go2_config"]]
        terrain_cfg = f"train_SAC/{eval_config['terrain_config']}"
        episodes = int(eval_config["episodes"])
        max_episode_steps = int(eval_config["max_episode_steps"])
        seed = int(eval_config["seed"])
    else:
        train_log_dir = os.path.join(current_path, "train_logs", policy)
        train_config_file = os.path.join(train_log_dir, "train_config.yaml")
        with open(train_config_file, "r", encoding="utf-8") as f:
            train_config = yaml.load(f, Loader=yaml.FullLoader)

        go2_cfg = [train_config["go2_task"], train_config["go2_config"]]
        terrain_cfg = f"train_SAC/train_logs/{policy}/terrain_config.yaml"
        episodes = int(eval_config["episodes"])
        max_episode_steps = int(train_config["max_episode_steps"])
        seed = 0

    return model, go2_cfg, terrain_cfg, episodes, max_episode_steps, seed


def main() -> None:
    current_path = os.path.dirname(os.path.abspath(__file__))
    cfg_file = os.path.join(current_path, "eval_config.yaml")
    with open(cfg_file, "r", encoding="utf-8") as f:
        eval_config = yaml.load(f, Loader=yaml.FullLoader)

    run_name = f"{str(eval_config['log_name'])}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = os.path.join(current_path, "eval_logs", run_name)
    print(f"[eval_SAC_sampling] run_dir: {log_dir}")
    model, go2_cfg, terrain_cfg, episodes, max_episode_steps, seed = _load_eval_context(eval_config, current_path)

    sampling_cfg = eval_config.get("sampling", {})
    uniform_weight = _clip_uniform_weight(float(sampling_cfg.get("uniform_weight", 0.0)))

    evaluate_policy_sampling(
        model=model,
        go2_cfg=go2_cfg,
        terrain_cfg=terrain_cfg,
        episodes=episodes,
        max_episode_steps=max_episode_steps,
        log_dir=log_dir,
        seed=seed,
        render=bool(eval_config.get("render", False)),
        uniform_weight=uniform_weight,
        save_non_failure_trajectories=bool(eval_config.get("save_non_failure_trajectories", False)),
    )


if __name__ == "__main__":
    main()

