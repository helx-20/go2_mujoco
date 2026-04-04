#!/usr/bin/env python3
"""
NADE-style test for go2_mujoco: optionally use a criticality model to weight failures.

If `--model_path` is provided, the script will attempt to load a PyTorch model
that maps observations to a criticality score in [0,1]. The script multiplies
failures by weights derived from that score to estimate a weighted failure count.
"""
import argparse
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import numpy as np
import math
import torch
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


def load_crit_model(path, device):
    if path is None:
        return None
    model = torch.load(path, map_location=device)
    # If saved state_dict, user should wrap in a small wrapper; here we accept a callable
    if isinstance(model, dict):
        # try to return state_dict and require user to provide model loader in future
        raise RuntimeError("Provided model file appears to be a state_dict. Please provide a callable model file.")
    return model


def run(args):
    go2_cfg = ("terrain", "go2.yaml")
    terrain_cfg = "terrain_config.yaml"

    trainer = TerrainTrainer(go2_cfg, terrain_cfg)
    env = TerrainGymEnv(trainer, max_episode_steps=args.max_steps)

    device = torch.device('cpu')
    crit_model = load_crit_model(args.model_path, device) if getattr(args, 'model_path', None) else None
    if crit_model is not None:
        crit_model.eval()

    n = args.episodes
    weighted_failures = []
    out_path = args.out
    save_interval = args.save_interval

    action_space = env.action_space

    # prepare discretization edges for actions: 10 bins per dimension (match nde_test_go2.py)
    action_edges = None
    if getattr(action_space, 'shape', None) and action_space.shape[0] > 0:
        try:
            low = np.asarray(action_space.low, dtype=np.float32)
            high = np.asarray(action_space.high, dtype=np.float32)
            # avoid zero-range
            high = np.where(high == low, low + 1.0, high)
            action_edges = [np.linspace(low[d], high[d], num=11) for d in range(low.shape[0])]
        except Exception:
            action_edges = None

    def sample_discrete_action():
        # returns (continuous_centers_array, discrete_bins_array)
        if action_edges is None:
            a = action_space.sample()
            return np.asarray(a, dtype=np.float32), None

        # sample bins uniformly per-dimension and map to continuous center values
        bins = np.random.randint(0, 10, size=action_space.shape)
        centers = np.zeros(action_space.shape, dtype=np.float32)
        flat_bins = np.asarray(bins).reshape(-1)
        for d in range(flat_bins.shape[0]):
            b = int(flat_bins[d])
            e = action_edges[d]
            centers.reshape(-1)[d] = 0.5 * (e[b] + e[b + 1])
        return centers, bins

    for i in range(n):
        obs, _ = env.reset()
        done = False
        while not done:
            # NADE: sample candidate actions and use criticality model to bias selection
            if getattr(action_space, 'shape', None) and action_space.shape[0] > 0:
                if crit_model is None:
                    # no criticality model: sample uniformly as before
                    centers_c, _ = sample_discrete_action()
                    action = np.asarray(centers_c, dtype=np.float32)
                else:
                    # Build full discrete action grid (10 bins per dim) and evaluate in parallel
                    if action_edges is None:
                        # fallback: sample candidate actions if edges unavailable
                        candidates = []
                        for _ in range(args.candidates):
                            centers_c, bins_c = sample_discrete_action()
                            candidates.append(np.asarray(centers_c, dtype=np.float32))
                        candidates_arr = np.stack(candidates, axis=0)
                    else:
                        D = int(action_space.shape[0])
                        grids = np.meshgrid(*[np.arange(10) for _ in range(D)], indexing='ij')
                        bins_flat = np.stack([g.reshape(-1) for g in grids], axis=1).astype(np.int64)
                        num_actions = bins_flat.shape[0]
                        centers = np.zeros((num_actions, D), dtype=np.float32)
                        for d in range(D):
                            e = action_edges[d]
                            b_idx = bins_flat[:, d]
                            centers[:, d] = 0.5 * (e[b_idx] + e[b_idx + 1])
                        candidates_arr = centers

                    # Evaluate crit_model in batch: try obs+action first, fallback to obs-only
                    with torch.no_grad():
                        t_obs = torch.from_numpy(obs.astype(np.float32)).to(device).unsqueeze(0).repeat(candidates_arr.shape[0], 1)
                        t_act = torch.from_numpy(np.asarray(candidates_arr, dtype=np.float32)).to(device)
                        try:
                            t_in = torch.cat([t_obs, t_act], dim=1)
                            outputs = crit_model(t_in)
                        except Exception:
                            try:
                                outputs = crit_model(t_obs)
                            except Exception:
                                outputs = torch.zeros(candidates_arr.shape[0], dtype=torch.float32, device=device)
                    scores = outputs.squeeze().cpu().numpy()
                    scores = np.clip(np.asarray(scores, dtype=np.float32), 0.0, 1.0)

                    # construct criticality distribution using threshold (binary) similar to MyLander
                    criticality = (scores > args.criticality_thresh).astype(float)
                    p_list = np.ones_like(criticality, dtype=float)
                    if criticality.sum() > 0:
                        criticality = criticality / criticality.sum()
                        p_list = p_list / p_list.sum()
                        pdf_array = (1.0 - args.epsilon) * criticality + args.epsilon * p_list
                        pdf_array = pdf_array / pdf_array.sum()
                    else:
                        pdf_array = p_list / p_list.sum()

                    # select according to pdf_array (or greedy)
                    if args.selection == 'greedy':
                        idx = int(np.argmax(scores))
                    else:
                        idx = int(np.random.choice(len(pdf_array), p=pdf_array))
                    action = np.asarray(candidates_arr[idx], dtype=np.float32)
                    weight = (p_list[idx] / pdf_array[idx]) if pdf_array[idx] > 0 else 1.0
            else:
                action = np.array([], dtype=np.float32)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated) or bool(truncated)
            if done and not truncated:
                crash = int(bool(info.get('fallen', False) or info.get('base_collision', False) or info.get('collided', False) or info.get('stuck', False)))
                # if crit_model provided, weight failure by model score at terminal obs if possible
                weight = 1.0
                if crit_model is not None:
                    try:
                        t = torch.from_numpy(obs.astype(np.float32)).to(device).unsqueeze(0)
                        out = crit_model(t)
                        weight = float(out.squeeze().cpu().item())
                    except Exception:
                        weight = 1.0
                weighted_failures.append(crash * max(0.0, min(1.0, weight)))

        if (i + 1) % args.log_interval == 0:
            Mean, RHF, Val = calculate_val(weighted_failures)
            print(f"episode {i+1}/{n}  samples={len(weighted_failures)}  weighted_mean={Mean[-1]:.6f}  RHF={RHF[-1]:.6f}")

        if (i + 1) % save_interval == 0:
            if out_path:
                np.save(os.path.join(out_path, f'nade_{args.worker_id}.npy'), np.array(weighted_failures, dtype=np.float32))

    Mean, RHF, Val = calculate_val(weighted_failures)
    print("DONE")
    print(f"samples={len(weighted_failures)}  weighted_mean={Mean[-1]:.6f}  RHF={RHF[-1]:.6f}")
    # final save
    if out_path:
        d = os.path.dirname(out_path)
        if d:
            os.makedirs(d, exist_ok=True)
        np.save(out_path, np.array(weighted_failures, dtype=np.float32))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker_id', type=int, default=0)
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--max_steps', type=int, default=1000)
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--model_path', type=str, default=None, help='Optional criticality model (callable)')
    parser.add_argument('--candidates', type=int, default=16, help='Number of candidate terrain actions to sample per decision')
    parser.add_argument('--selection', choices=['prob', 'greedy'], default='prob', help='Selection mode: probability sampling or greedy')
    parser.add_argument('--out', type=str, default='results/nade_weighted.npy', help='Path to save weighted failures numpy array')
    parser.add_argument('--save_interval', type=int, default=200, help='Save results every N episodes at log interval')
    parser.add_argument('--epsilon', type=float, default=0.5, help='epsilon mixing weight for criticality/pdf')
    parser.add_argument('--criticality_thresh', type=float, default=0.5, help='threshold to binarize criticality scores')
    args = parser.parse_args()
    # ensure output directory exists (treat --out as directory to mirror nde_test_go2.py behavior)
    if args.out:
        os.makedirs(args.out, exist_ok=True)
    run(args)
