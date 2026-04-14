#!/usr/bin/env python3
"""
d2rl-style test for go2_mujoco: optionally use a criticality model to weight failures.

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
from criticality.utils.criticality_model import SimpleClassifier

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
    model = SimpleClassifier(input_dim=56)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.to(device)
    return model


def compute_epsilon(observation, epsilon_model, device=torch.device('cpu')):
    obs = torch.reshape(torch.tensor(observation,device=device), (1,len(observation))).to(device)
    out = epsilon_model({"obs":obs},[torch.tensor([0.0]).to(device)],torch.tensor([1]).to(device))
    action = np.clip((float(out[0][0][0]) + 1) * (0.999 - 0.001) / 2 + 0.001, 0.001, 0.999)
    return action


def run(args):
    go2_cfg = ("terrain", "go2.yaml")
    terrain_cfg = "terrain_config.yaml"

    trainer = TerrainTrainer(go2_cfg, terrain_cfg)
    env = TerrainGymEnv(trainer, max_episode_steps=args.max_steps)

    device = torch.device('cpu')
    crit_model = load_crit_model(args.model_path, device) if getattr(args, 'model_path', None) else None
    if crit_model is not None:
        crit_model.eval()
    else:
        raise RuntimeError("Criticality model is required for d2rl.")
    
    epsilon_model = torch.load(args.epsilon_model_path, map_location=device)
    epsilon_model.eval()

    n = args.episodes
    weighted_failures = []
    out_path = args.out
    save_interval = args.save_interval

    action_space = env.action_space

    # prepare discretization edges for actions: 10 bins per dimension (match nde_test_go2.py)
    low = np.asarray(action_space.low, dtype=np.float32)
    high = np.asarray(action_space.high, dtype=np.float32)
    # avoid zero-range
    high = np.where(high == low, low + 1.0, high)
    action_edges = [np.linspace(low[d], high[d], num=11) for d in range(low.shape[0])]
    
    # Build full discrete action grid (10 bins per dim) and evaluate in parallel
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

    criticality_data_all = []
    for i in range(n):
        obs, _ = env.reset()
        done = False
        total_weight = 1.0
        steps = 0

        # per-episode buffers
        ep_obs = []
        ep_actions = []

        while not done:
            steps += 1

            # Evaluate crit_model in batch: try obs+action first, fallback to obs-only
            with torch.no_grad():
                t_obs = torch.from_numpy(obs.astype(np.float32)).to(device).unsqueeze(0).repeat(candidates_arr.shape[0], 1)
                t_act = torch.from_numpy(np.asarray(candidates_arr, dtype=np.float32)).to(device)
                t_in = torch.cat([t_obs, t_act], dim=1)
                outputs = crit_model(t_in)
            scores = torch.nn.functional.softmax(outputs, dim=1)[:, 1].squeeze().cpu().numpy()

            # construct criticality distribution
            if args.criticality_thresh is not None:
                criticality = (scores > args.criticality_thresh).astype(float)
            else:
                criticality = scores
            p_list = np.ones_like(criticality, dtype=float)
            p_list = p_list / p_list.sum()
            if np.max(criticality) > args.criticality_max_thresh: # or np.sum(criticality) > 200 * args.criticality_max_thresh:
                criticality = criticality / criticality.sum()
                epsilon = compute_epsilon(obs, epsilon_model)
                pdf_array = (1.0 - epsilon) * criticality + epsilon * p_list
            else:
                pdf_array = p_list

            pdf_array = pdf_array / pdf_array.sum()  # ensure normalized
            idx = int(np.random.choice(len(pdf_array), p=pdf_array))
            action = np.asarray(candidates_arr[idx], dtype=np.float32)
            weight = p_list[idx] / pdf_array[idx]

            if total_weight * weight < args.min_weight:
                pdf_array = p_list  # fallback to uniform if weight too small
                idx = int(np.random.choice(len(pdf_array), p=pdf_array))
                action = np.asarray(candidates_arr[idx], dtype=np.float32)
                weight = 1.0
            total_weight *= weight

            if args.criticality_out is not None:
                ep_obs.append(obs)
                ep_actions.append(action)

            next_obs, reward, terminated, truncated, info = env.step(action)
            obs = next_obs
            done = bool(terminated) or bool(truncated) or bool(info.get('fallen', False) or info.get('collided', False) or info.get('base_collision', False) or info.get('thigh_collision', False) or info.get('stuck', False))

        # episode finished; determine crash/failure from last step's info
        crash = int(bool(info.get('fallen', False) or info.get('collided', False) or info.get('base_collision', False) or info.get('thigh_collision', False) or info.get('stuck', False)))
        weighted_failures.append(crash * total_weight)
        print(f"episode {i+1}/{n} steps={steps} crash={crash>0} weight={total_weight:.6f}")

        if args.criticality_out is not None and crash:
            label = 1
            crash_type = 'fallen' if info.get('fallen', False) else 'collided' if info.get('collided', False) else 'base_collision' if info.get('base_collision', False) else 'thigh_collision' if info.get('thigh_collision', False) else 'stuck' if info.get('stuck', False) else 'none'
            criticality_data_all.append({'obs': ep_obs, 'actions': ep_actions, 'label': label, 'crash_type': crash_type})

        if (i + 1) % args.log_interval == 0:
            Mean, RHF, Val = calculate_val(weighted_failures)
            print(f"episode {i+1}/{n}  samples={len(weighted_failures)}  weighted_mean={Mean[-1]:.6f}  RHF={RHF[-1]:.6f}")

        if (i + 1) % save_interval == 0:
            if out_path:
                np.save(os.path.join(out_path, f'd2rl_{args.worker_id}.npy'), np.array(weighted_failures, dtype=np.float32))
            if args.criticality_out is not None:
                try:
                    np.save(os.path.join(args.criticality_out, f'd2rl_{args.worker_id}.npy'), np.array(criticality_data_all))
                    print(f'Wrote criticality data to {args.criticality_out} (samples={len(criticality_data_all)})')
                except Exception as e:
                    print('Failed to save criticality data:', e)

    Mean, RHF, Val = calculate_val(weighted_failures)
    print("DONE")
    print(f"samples={len(weighted_failures)}  weighted_mean={Mean[-1]:.6f}  RHF={RHF[-1]:.6f}")
    # final save
    if out_path:
        np.save(os.path.join(out_path, f'd2rl_{args.worker_id}.npy'), np.array(weighted_failures, dtype=np.float32))

    if args.criticality_out is not None:
        try:
            np.save(os.path.join(args.criticality_out, f'd2rl_{args.worker_id}.npy'), np.array(criticality_data_all))
            print(f'Wrote criticality data to {args.criticality_out} (samples={len(criticality_data_all)})')
        except Exception as e:
            print('Failed to save criticality data:', e)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker_id', type=int, default=0)
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--max_steps', type=int, default=40)
    parser.add_argument('--log_interval', type=int, default=10)
    # parser.add_argument('--model_path', type=str, default='criticality/stage1/model/stage1_criticality_best_new_1.pt', help='Optional criticality model')
    parser.add_argument('--model_path', type=str, default='criticality/stage1_plus/model/stage1_plus_criticality_best_new_3.pt', help='Optional criticality model')
    # parser.add_argument('--model_path', type=str, default='criticality/stage2/model/stage2_new_1_epoch5950.pt', help='Optional criticality model')
    parser.add_argument('--epsilon_model_path', type=str, default='epsilon/model/epsilon_model.pt')
    parser.add_argument('--out', type=str, default='results/d2rl/', help='Path to save weighted failures numpy array')
    parser.add_argument('--criticality_out', type=str, default=None, help='Optional path to save criticality dataset (obs, actions, labels)')
    parser.add_argument('--save_interval', type=int, default=5, help='Save results every N episodes at log interval')
    parser.add_argument('--criticality_thresh', type=float, default=None, help='threshold to binarize criticality scores')
    parser.add_argument('--criticality_max_thresh', type=float, default=3e-1, help='threshold to determine if crit scores are meaningful for weighting')
    parser.add_argument('--min_weight', type=float, default=1e-3, help='minimum weight to apply to failures')
    args = parser.parse_args()
    print('args:', args)
    # ensure output directory exists
    if args.out:
        os.makedirs(args.out, exist_ok=True)
    if args.criticality_out:
        os.makedirs(args.criticality_out, exist_ok=True)
    run(args)
