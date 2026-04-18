#!/usr/bin/env python3
"""
Simple NDE-style test for go2_mujoco: run many deterministic episodes without terrain modifications
and compute running failure statistics (mean, relative half-width).
"""
import argparse
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import numpy as np
import math
from scipy.stats import norm

from deploy_mujoco.terrain_trainer import TerrainTrainer, TerrainGymEnv


alpha = 0.05
z = norm.isf(q=alpha)


def calculate_val(the_list):
    Mean = []
    Relative_half_width = []
    Var = []
    var_old = 0
    mean_old = 0
    for i in range(len(the_list)):
        if math.isnan(the_list[i]) or math.isinf(the_list[i]):
            the_list[i] = 0.0
        n = i + 1
        mean_new = mean_old + (the_list[i] - mean_old) / n
        Mean.append(mean_new)
        var_new = (n - 1) * var_old / n + (n - 1) * (the_list[i] - mean_old) ** 2 / (n * n)
        Var.append(1.96 * (np.sqrt(var_new / n)))
        Relative_half_width.append(z * (np.sqrt(var_new / n) / (mean_new + 1e-30)))
        var_old = var_new
        mean_old = mean_new
    return Mean, Relative_half_width, Var


def run(args):
    # go2_config_file: (foldername, cfg filename) relative to deploy_mujoco
    go2_cfg = ("terrain", "go2.yaml")
    terrain_cfg = "terrain_config.yaml"

    trainer = TerrainTrainer(go2_cfg, terrain_cfg)
    env = TerrainGymEnv(trainer, max_episode_steps=args.max_steps)

    n = args.episodes
    crashes = []
    out_path = args.out
    crit_out = getattr(args, 'criticality_out', None)

    action_space = env.action_space
    # prepare discretization edges for actions: 10 bins per dimension
    low = np.asarray(action_space.low, dtype=np.float32)
    high = np.asarray(action_space.high, dtype=np.float32)
    # avoid zero-range
    high = np.where(high == low, low + 1.0, high)
    action_edges = [np.linspace(low[d], high[d], num=11) for d in range(low.shape[0])]

    # global buffers for criticality training data (across episodes)
    crit_data_all = []

    for i in range(n):
        obs, _ = env.reset()
        done = False
        steps = 0

        # per-episode buffers
        ep_obs = []
        ep_actions = []

        while not done:
            steps += 1
            # NDE: uniformly sample a terrain action from the environment action space
            # sample bins uniformly
            bins = np.random.randint(0, 10, size=action_space.shape)
            # map bins to continuous center values
            centers = np.zeros(action_space.shape, dtype=np.float32)
            flat_bins = np.asarray(bins).reshape(-1)
            for d in range(flat_bins.shape[0]):
                b = int(flat_bins[d])
                e = action_edges[d]
                centers.reshape(-1)[d] = 0.5 * (e[b] + e[b+1])
            action = centers
            # store discrete bins as action representation
            action = np.asarray(action, dtype=np.float32)

            # record observation and action for criticality dataset (obs before applying action)
            if crit_out is not None:
                ep_obs.append(obs)
                ep_actions.append(action)

            next_obs, reward, terminated, truncated, info = env.step(action)
            obs = next_obs
            done = bool(terminated) or bool(truncated) or bool(info.get('fallen', False) or info.get('collided', False) or info.get('base_collision', False) or info.get('thigh_collision', False) or info.get('stuck', False))

        # episode finished; determine crash/failure from last step's info
        crash = int(bool(info.get('fallen', False) or info.get('collided', False) or info.get('base_collision', False) or info.get('thigh_collision', False) or info.get('stuck', False)))
        crashes.append(crash)
        crash_type = 'fallen' if info.get('fallen', False) else 'collided' if info.get('collided', False) else 'base_collision' if info.get('base_collision', False) else 'thigh_collision' if info.get('thigh_collision', False) else 'stuck' if info.get('stuck', False) else 'none'
        print(f"episode {i+1}/{n} steps={steps} crash={crash}")

        # label and append per-episode data to global criticality buffers
        if crit_out is not None:
            label = int(crash)
            crit_data_all.append({'obs': ep_obs, 'actions': ep_actions, 'label': label, 'crash_type': crash_type})

        if (i + 1) % args.log_interval == 0:
            Mean, RHF, Val = calculate_val(crashes)
            print(f"episode {i+1}/{n}  samples={len(crashes)}  Mean={Mean[-1]:.6f}  RHF={RHF[-1]:.6f}")

        if (i + 1) % args.save_interval == 0:
            if out_path:
                np.save(os.path.join(out_path, f'nde_{args.worker_id}.npy'), np.array(crashes, dtype=np.float32))
            # also save criticality buffers periodically
            if crit_out is not None:
                try:
                    np.save(os.path.join(crit_out, f'nde_{args.worker_id}.npy'), np.array(crit_data_all))
                    print(f'Wrote criticality data to {crit_out} (samples={len(crit_data_all)})')
                except Exception as e:
                    print('Failed to save criticality data:', e)

    # final stats
    Mean, RHF, Val = calculate_val(crashes)
    print("DONE")
    print(f"samples={len(crashes)}  Mean={Mean[-1]:.6f}  RHF={RHF[-1]:.6f}")

    # final save
    if out_path:
        np.save(os.path.join(out_path, f'nde_{args.worker_id}.npy'), np.array(crashes, dtype=np.float32))

    # final save criticality data
    if crit_out is not None:
        try:
            np.save(os.path.join(crit_out, f'nde_{args.worker_id}.npy'), np.array(crit_data_all))
            print(f'Wrote criticality data to {crit_out} (samples={len(crit_data_all)})')
        except Exception as e:
            print('Failed to save criticality data:', e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker_id', type=int, default=0)
    parser.add_argument('--episodes', type=int, default=2000)
    parser.add_argument('--max_steps', type=int, default=40)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--out', type=str, default='results/nde', help='Path to save crashes numpy array')
    parser.add_argument('--save_interval', type=int, default=10, help='Save results every N episodes at log interval')
    parser.add_argument('--criticality_out', type=str, default="data/nde", help='Optional path to save criticality dataset (obs, actions, labels)')
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.criticality_out:
        os.makedirs(args.criticality_out, exist_ok=True)
    run(args)
