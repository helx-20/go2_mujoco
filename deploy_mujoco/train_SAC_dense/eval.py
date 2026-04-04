import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

import csv
from datetime import datetime
import pickle
import shutil
import numpy as np
from stable_baselines3 import SAC
from deploy_mujoco.terrain_trainer import TerrainTrainer, TerrainGymEnv
import yaml


def _build_log_paths(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    return {
        "log_dir": log_dir,
        "pkl": os.path.join(log_dir, "collision_failures.pkl"),
        "non_failure_pkl": os.path.join(log_dir, "non_failure_trajectories.pkl"),
        "csv": os.path.join(log_dir, "failure_summary.csv"),
    }


def _append_csv_row(csv_path, row):
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


def evaluate_policy(
    model,
    go2_cfg,
    terrain_cfg,
    episodes,
    max_episode_steps,
    log_dir,
    seed,
    render,
    save_non_failure_trajectories=False,
):

    current_path = os.path.dirname(os.path.realpath(__file__))

    trainer = TerrainTrainer(go2_cfg, terrain_cfg)
    if not trainer.render and render:
        trainer.render = True
        trainer.start_viewer()

    env = TerrainGymEnv(trainer, max_episode_steps=max_episode_steps)

    if model is None:
        rng = np.random.default_rng(seed)
    else:
        rng = None

    paths = _build_log_paths(log_dir)

    go2_cfg_file = os.path.join(current_path, "../", go2_cfg[0], "configs", go2_cfg[1])
    terrain_cfg_file = os.path.join(current_path, "../", terrain_cfg)
    # 复制文件到logs
    os.makedirs(paths["log_dir"], exist_ok=True)
    shutil.copy2(go2_cfg_file, paths["log_dir"])
    shutil.copy2(terrain_cfg_file, paths["log_dir"])

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

    for ep in range(episodes):
        obs, info = env.reset()
        ep_chain = []
        has_failure = False
        has_collision = False
        has_fall = False
        has_base_collision = False
        has_thigh_collision = False
        has_stuck = False

        for step_idx in range(max_episode_steps):

            print(f"Episode {ep+1}/{episodes}, Step {step_idx+1}/{max_episode_steps}", end="\r")

            if model is None:
                action = rng.uniform(-1.0, 1.0, size=env.action_space.shape).astype(np.float32)
            else:
                action, _ = model.predict(obs, deterministic=False)
            next_obs, reward, terminated, truncated, info = env.step(action)

            # info_dict = {
            #     "fallen": bool(info.get("fallen", False)),
            #     "collided": bool(info.get("collided", False)),
            #     "base_collision": bool(info.get("base_collision", False)),
            #     "thigh_collision": bool(info.get("thigh_collision", False)),
            #     "terrain_reward": float(info.get("terrain_reward", 0.0)),
            #     "stuck": bool(info.get("stuck", False)),
            # }
            info_dict = info  # info中包含所有奖励项，包括未加权值

            transition = {
                "obs": np.asarray(obs, dtype=np.float32).tolist(),
                "action": np.asarray(action, dtype=np.float32).tolist(),
                "reward": float(reward),
                "next_obs": np.asarray(next_obs, dtype=np.float32).tolist(),
                "done": bool(terminated or truncated),
                "terrain_control": np.asarray(action, dtype=np.float32).tolist(),
                "info": info_dict,
            }
            ep_chain.append(transition)

            has_collision = has_collision  # or info_dict["collided"]
            has_fall = has_fall or info_dict["fallen"]
            has_base_collision = has_base_collision or info_dict["base_collision"]
            has_thigh_collision = has_thigh_collision or info_dict["thigh_collision"]
            has_stuck = has_stuck or info_dict["stuck"]
            has_failure = has_collision or has_fall or has_stuck or has_base_collision or has_thigh_collision

            obs = next_obs
            if terminated or truncated:
                break

        summary["episodes_evaluated"] += 1
        if has_failure:
            summary["total_failures"] += 1
        if has_collision:
            summary["collision_failures"] += 1
        if has_fall:
            summary["fall_failures"] += 1
        if has_base_collision:
            summary["base_collision_failures"] += 1
        if has_thigh_collision:
            summary["thigh_collision_failures"] += 1
        if has_stuck:
            summary["stuck_failures"] += 1

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


def main():
    current_path = os.path.dirname(os.path.abspath(__file__))
    eval_config_file = "eval_config.yaml"
    with open(f"{current_path}/{eval_config_file}", "r", encoding="utf-8") as f:
        eval_config = yaml.load(f, Loader=yaml.FullLoader)

    current_path = os.path.dirname(os.path.realpath(__file__))
    run_name = f"{str(eval_config['log_name'])}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = os.path.join(current_path, "eval_logs", run_name)
    print(f"[eval_SAC_dense] run_dir: {log_dir}")

    policy = eval_config['policy']
    checkpoint = eval_config['checkpoint']

    if policy == "random":
        print("Evaluating random policy...")
        model = None
    else:
        print(f"Evaluating policy from {policy}...")
        model_path = str(os.path.join(current_path, "train_logs", policy, checkpoint))
        model = SAC.load(model_path)

    if policy == "random" or eval_config["use_curr_force"]:
        go2_cfg = [eval_config['go2_task'], eval_config['go2_config']]
        terrain_cfg = f"train_SAC_dense/{eval_config['terrain_config']}"
        episodes = eval_config['episodes']
        max_episode_steps = eval_config['max_episode_steps']
        seed = eval_config['seed']
    else:
        current_path = os.path.dirname(os.path.realpath(__file__))
        train_log_dir = os.path.join(current_path, "train_logs", policy)
        train_config_file = os.path.join(train_log_dir, "train_config.yaml")

        with open(train_config_file, "r", encoding="utf-8") as f:
            train_config = yaml.load(f, Loader=yaml.FullLoader)

        go2_cfg = [train_config['go2_task'], train_config['go2_config']]
        terrain_cfg = f"train_SAC_dense/train_logs/{policy}/terrain_config.yaml"
        episodes = eval_config['episodes']
        max_episode_steps = train_config['max_episode_steps']
        seed = 0

    render = eval_config["render"]
    save_non_failure_trajectories = bool(eval_config.get("save_non_failure_trajectories", False))

    evaluate_policy(
        model=model,
        go2_cfg=go2_cfg,
        terrain_cfg=terrain_cfg,
        episodes=episodes,
        max_episode_steps=max_episode_steps,
        log_dir=log_dir,
        seed=seed,
        render=render,
        save_non_failure_trajectories=save_non_failure_trajectories,
    )


if __name__ == "__main__":
    main()
