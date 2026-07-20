#!/usr/bin/env python3
"""Train Go2 controller policy in MuJoCo using stable-baselines3 PPO.

This script aims to reproduce the training style of the original
`legged_gym/scripts/train.py` call (PPO, many environments, seeds,
logging). It trains a policy that maps controller observations to
controller actions (num_actions) and saves the trained model.
"""
import os
import sys
import argparse
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import yaml
import numpy as np
# from stable_baselines3 import PPO
from training.utils.ppo import PPO
import types
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.logger import configure
# from stable_baselines3.common.buffers import RolloutBuffer
from training.utils.buffers import RolloutBuffer
from training.utils.train_env import TrainEnv
from training.utils.test_env import TestEnv, TerrainGymEnv

# Silence Gym -> Gymnasium migration warning and noisy gym/gymnasium loggers
import warnings
import logging
warnings.filterwarnings("ignore", message=".*Please upgrade to Gymnasium.*", category=UserWarning)
logging.getLogger("gym").setLevel(logging.ERROR)
logging.getLogger("gymnasium").setLevel(logging.ERROR)


def make_env_fn(normal_policy, max_episode_steps=1000, nade=False, criticality_model=None, critical_threshold=0.5):
    def _thunk():
        # Create a lightweight dummy policy to satisfy TestEnv/Go2Controller
        config_file_path = "go2_training.yaml"
        terrain_cfg = "terrain_config.yaml"
        trainer = TestEnv(policy=normal_policy, config_file_path=config_file_path, terrain_config_file=terrain_cfg, critical_threshold=critical_threshold)
        env = TrainEnv(trainer=trainer, max_episode_steps=max_episode_steps, nade=nade, criticality_model=criticality_model)
        return Monitor(env)

    return _thunk

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rl_device', type=str, default='cuda:2')
    parser.add_argument('--max_steps', type=int, default=30)
    parser.add_argument('--n_eval_episodes', type=int, default=1, help='Number of episodes per evaluation')
    parser.add_argument('--out', type=str, default='training/models/')
    parser.add_argument('--run_name', type=str, default='run_offline_round8', help='Subdirectory name for this training run')
    parser.add_argument('--pretrain', type=str, default='training/models/actor_init.zip',
                        help='Path to a pretrained PyTorch model or SB3 .zip to initialize normal policy (default uses training/models/actor_init.zip)')
    parser.add_argument('--criticality_model_path', type=str, default='criticality/stage1/model/stage1_criticality_best_new_3.pt', help='Path to criticality model')
    # parser.add_argument('--initial', default='training/models/actor_init.zip')
    parser.add_argument('--initial', default='training/models/run_offline_round7/best.policy.pt')
    parser.add_argument('--log_std', type=float, default=None,
                        help='Initial log_std for policy distribution (per-dim, trainable). '
                             '-1 (std ~= 0.37) matches the collection-time setting and keeps the '
                             'policy sharp enough for AWAC gradient on the mean to be strong. '
                             'Extreme logp on outlier data points is bounded by max_grad_norm clipping.')
    parser.add_argument('--log_std_min', type=float, default=-4.0, help='Lower clamp on trainable log_std during loss computation.')
    parser.add_argument('--log_std_max', type=float, default=1.0, help='Upper clamp on trainable log_std during loss computation.')
    parser.add_argument('--dataset', type=list, default=['/mnt/mnt1/linxuan/go2_data/data/training/round7', '/mnt/mnt1/linxuan/go2_data/data/training/round7_append', '/mnt/mnt1/linxuan/go2_data/data/training/round7', '/mnt/mnt1/linxuan/go2_data/data/training/round7_thr05', '/mnt/mnt1/linxuan/go2_data/data/training/round8', '/mnt/mnt1/linxuan/go2_data/data/training/round8_append'], help='Path to offline dataset directory')
    parser.add_argument('--offline_epochs', type=int, default=30, help='Epochs for offline training')
    parser.add_argument('--offline_batch_size', type=int, default=2048)
    parser.add_argument('--offline_lr', type=float, default=1e-4)
    parser.add_argument('--train_value_net_only', action='store_true', help='Only train value net during offline training (policy net weights will be frozen)')
    parser.add_argument('--use_initial_optimizer', action='store_true', help='Whether to load optimizer state from initial .pt file if available (ignored if initial is SB3 .zip)')
    # AWAC/AWR + stability
    parser.add_argument('--awac_beta', type=float, default=0.5,
                        help='AWAC/AWR temperature. Smaller -> sharper exploitation of high-adv samples; larger -> more uniform. '
                             'beta=1 keeps meaningful discrimination between good/bad samples; '
                             'combined_weight_max already caps outliers, so no need to soften further.')
    parser.add_argument('--awac_weight_max', type=float, default=20.0,
                        help='Upper clip for exp(adv/beta) to prevent exploding weights.')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Max grad norm for clipping.')
    parser.add_argument('--bc_coef', type=float, default=0.3,
                        help='Behavior-cloning regularizer weight (KL-to-behavior proxy under fixed log_std). Set 0 to disable.')
    parser.add_argument('--value_coef', type=float, default=50.0,
                        help='Weight for value regression loss during joint training. '
                             'Needs to be large because policy_loss is O(10) while value_loss is O(0.04).')
    # Value warmup / validation / early stopping
    parser.add_argument('--value_warmup_epochs', type=int, default=2,
                        help='Train value net only for the first N epochs, then jointly train policy+value. Ignored if --train_value_net_only.')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Fraction of offline dataset held out for validation. 0 disables validation.')
    parser.add_argument('--early_stop_patience', type=int, default=3,
                        help='Early stop after N non-improving val epochs. 0 disables.')
    # Robustness against outlier samples in value regression & advantage weighting
    parser.add_argument('--reset_value_net', action='store_true', default=False,
                        help='Re-init value_net after copying pretrained weights. Disabled by default '
                             'because actor_init.zip never trained its value_net (it was created '
                             'purely for action pretraining), so weights are already random -- '
                             'overwriting them with orthogonal init adds nothing.')
    parser.add_argument('--value_loss', type=str, default='huber', choices=['mse', 'huber'],
                        help='Value regression loss. Huber caps gradient on outliers.')
    parser.add_argument('--huber_delta', type=float, default=1.0,
                        help='Huber loss delta (|error|>delta switches from quadratic to linear). '
                             'For returns in [-1,0], delta=1 behaves like MSE for most samples and '
                             'only clips rare extreme errors -- keeps value gradient strong enough to learn.')
    parser.add_argument('--combined_weight_max', type=float, default=20.0,
                        help='Cap on (awac_w * b_weights) per sample to prevent outlier domination.')
    parser.add_argument('--use_weighted_sampler', default=True,
                        help='Use WeightedRandomSampler so each batch over-represents high-weight (rare critical) '
                             'samples. Reduces gradient variance on the important tail. When enabled, in-loss '
                             'b_weights reweighting is removed (sampler already encodes the distribution).')
    parser.add_argument('--filter_awac', default=True,
                        help='In filter-BC mode, additionally weight kept samples by exp(adv/beta). This reuses '
                             'the value function to prioritize the best non-crash samples without imitating crashes.')
    parser.add_argument('--oob_coef', type=float, default=5,
                        help='Coefficient on the out-of-bounds soft penalty on pred_act. Zero-gradient inside '
                             '[-oob_bound, +oob_bound], quadratic outside. Keeps MLP extrapolation in check on '
                             'OOD states without hard-clipping gradients inside the data range. Set 0 to disable.')
    parser.add_argument('--oob_bound', type=float, default=5.0,
                        help='Action magnitude above which the OOB penalty engages. Dataset actions are within '
                             '[-5, 5], so 5 is the natural choice.')
    parser.add_argument('--crash_coef', type=float, default=3.0,
                        help='Weight for action-space hinge repulsion on crash samples. '
                             'Loss = relu(crash_margin - ||pred_act - crash_act||^2), bounded and zero once '
                             'policy is far enough from crash actions. Only active when filter_bc=True.')
    parser.add_argument('--crash_margin', type=float, default=1.0,
                        help='L2-squared margin for crash repulsion hinge. Policy stops being penalized once '
                             '||pred_act - crash_act||^2 >= crash_margin. With 12-dim actions ~N(0,1.87^2), '
                             'margin=4 corresponds to mean per-dim offset of ~0.58 (about 0.3 std).')
    # filter-BC baseline: skip AWAC entirely, BC only on samples with return above threshold.
    # Motivation: with sparse binary rewards (return in {-1, 0}) and small data, AWAC's
    # advantage-weighted logp has low SNR. Filter-BC reduces the signal to "imitate only
    # non-crash trajectories" which is a stronger baseline under these conditions.
    parser.add_argument('--filter_bc', default=True,
                        help='Replace AWAC policy loss with filter-BC: BC only on samples with b_ret > --filter_return_threshold. '
                             'Value net still trains normally so it is available for downstream PPO. '
                             'Default True; pass --no-filter_bc to fall back to AWAC.')
    parser.add_argument('--filter_return_threshold', type=float, default=-0.01,
                        help='Keep samples with b_ret > this threshold for filter-BC. '
                             'With returns in {-1, 0}, default -0.5 keeps all non-crash samples and drops crashes.')
    parser.add_argument('--debug_first_batch', action='store_true', default=True,
                        help='Print min/max/percentiles of dataset tensors and first batch stats.')
    args = parser.parse_args()
    args.out = os.path.join(args.out, args.run_name)
    print(args)

    os.makedirs(args.out, exist_ok=True)

    # If a pretrained SB3 .zip is provided, load it only for weight extraction (no env attachment)
    # We will always create a fresh PPO model below and copy weights into it to avoid carrying over
    # stale internal training state (rollout buffer, optim state, etc.) from the checkpoint.
    sb3_pretrain_model = None
    if args.pretrain.endswith('.zip'):
        from stable_baselines3 import PPO as SB3PPO
        print('Found SB3 pretrained .zip at', args.pretrain, '- will map its policy into a fresh model')
        sb3_pretrain_model = SB3PPO.load(args.pretrain, device='cpu')

    class PolicyOnlyWrapper(torch.nn.Module):
        def __init__(self, net_pi, act_net):
            super().__init__()
            self.net_pi = net_pi
            self.act_net = act_net

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            latent = self.net_pi(x)
            actions = self.act_net(latent)
            return actions
        
    # create vec env
    # create envs with rendering disabled by default
    pretrain_wrapper = PolicyOnlyWrapper(sb3_pretrain_model.policy.mlp_extractor.policy_net.to('cpu').eval(), sb3_pretrain_model.policy.action_net.to('cpu').eval()).cpu()

    # load criticality model
    from criticality.utils.criticality_model import SimpleClassifier
    criticality_model = SimpleClassifier(input_dim=56)
    criticality_model.load_state_dict(torch.load(args.criticality_model_path, map_location='cpu', weights_only=False))
    criticality_model.to('cpu').eval()
    
    env_fns = [make_env_fn(pretrain_wrapper)]
    vec_env = DummyVecEnv(env_fns)

    # Create a fresh model and try to copy compatible weights if available
    hidden_sizes = [512, 256, 128]
    policy_kwargs = {'net_arch': dict(pi=hidden_sizes, vf=hidden_sizes), 'activation_fn': torch.nn.ELU}
    # Create model with CLI-configured hyperparameters
    model = PPO(
        'MlpPolicy', vec_env, verbose=1, device=args.rl_device,
        policy_kwargs=policy_kwargs,
    )

    if args.initial and os.path.exists(args.initial):
        # prefer already-loaded sb3_pretrain_model if available
        if args.initial.endswith('.zip'):
            initial_model = sb3_pretrain_model if sb3_pretrain_model is not None else __import__('stable_baselines3').PPO.load(args.initial, device='cpu')
            src_sd = initial_model.policy.state_dict()
        elif args.initial.endswith('.pt'):
            state_dict = torch.load(args.initial, map_location='cpu', weights_only=False)
            src_sd = state_dict['policy_state_dict']

        # Source and destination state dicts
        policy_sd = model.policy.state_dict()

        matched = {}
        used_src = set()

        # 1) Exact name + shape matches
        for src_k, src_v in src_sd.items():
            if src_k in policy_sd and tuple(policy_sd[src_k].shape) == tuple(src_v.shape):
                policy_sd[src_k].copy_(src_v)
                matched[src_k] = src_k
                used_src.add(src_k)

        # 2) Suffix (last token) + shape matches for remaining dst keys
        for dst_k in list(policy_sd.keys()):
            if dst_k in matched:
                continue
            dst_shape = tuple(policy_sd[dst_k].shape)
            dst_suffix = dst_k.split('.')[-1]
            for src_k, src_v in src_sd.items():
                if src_k in used_src:
                    continue
                if src_k.split('.')[-1] == dst_suffix and tuple(src_v.shape) == dst_shape:
                    policy_sd[dst_k].copy_(src_v)
                    matched[dst_k] = src_k
                    used_src.add(src_k)
                    break

        # 3) Shape-only matching for any remaining dst keys (first-fit)
        for dst_k in list(policy_sd.keys()):
            if dst_k in matched:
                continue
            dst_shape = tuple(policy_sd[dst_k].shape)
            for src_k, src_v in src_sd.items():
                if src_k in used_src:
                    continue
                if tuple(src_v.shape) == dst_shape:
                    policy_sd[dst_k].copy_(src_v)
                    matched[dst_k] = src_k
                    used_src.add(src_k)
                    break

        model.policy.load_state_dict(policy_sd)
        print(f'Initialized policy from {args.initial} — matched {len(matched)} tensors')

        # Re-init value head only: pretrained scalar head was trained against a different
        # return scale, so its bias/scaling is wrong. Keep the shared feature extractor
        # (mlp_extractor.value_net) so warmup doesn't have to re-learn representations from scratch.
        if args.reset_value_net:
            import torch.nn as _nn
            reset_count = 0
            for m in model.policy.value_net.modules():
                if isinstance(m, _nn.Linear):
                    _nn.init.orthogonal_(m.weight, gain=0.01)
                    if m.bias is not None:
                        _nn.init.zeros_(m.bias)
                    reset_count += 1
            print(f'Reset {reset_count} Linear layers in value head (mlp_extractor.value_net features preserved)')

    if args.log_std is not None:
        try:
            with torch.no_grad():
                model.policy.log_std.fill_(args.log_std)
        except Exception:
            pass

    def load_offline_dataset(data_dirs):
        all_obs, all_acts, all_returns, all_weights, all_log_prob = np.zeros((0, 48)), np.zeros((0, 12)), np.zeros((0,)), np.zeros((0,)), np.zeros((0,))
        gamma = getattr(model, 'gamma', 0.99)
        for data_dir in data_dirs:
            all_data_path = os.path.join(data_dir, 'all_data_unified_weight.npy')
            if os.path.exists(all_data_path):
                data = np.load(all_data_path, allow_pickle=True).item()
                obs = np.array(data['obs'], dtype=np.float32)
                acts = np.array(data['actions'], dtype=np.float32)
                returns = np.array(data['returns'], dtype=np.float32)
                weights = np.array(data['weights'], dtype=np.float32)
                log_prob = np.array(data['log_prob'], dtype=np.float32)

                all_obs = np.concatenate([all_obs, obs])
                all_acts = np.concatenate([all_acts, acts])
                all_returns = np.concatenate([all_returns, returns])
                all_weights = np.concatenate([all_weights, weights])
                all_log_prob = np.concatenate([all_log_prob, log_prob])
            else:
                tmp_obs, tmp_acts, tmp_returns, tmp_weights, tmp_log_prob = np.zeros((0, 48)), np.zeros((0, 12)), np.zeros((0,)), np.zeros((0,)), np.zeros((0,))
                for filename in os.listdir(data_dir):
                    if filename.endswith('.npy') and not filename.startswith('all'):
                        path = os.path.join(data_dir, filename)
                        try:
                            data = np.load(path, allow_pickle=True).item()
                        except Exception as e:
                            continue
                        obs = np.array(data['obs'], dtype=np.float32)
                        acts = np.array(data['actions'], dtype=np.float32)
                        rews = np.array(data['rewards'], dtype=np.float32)
                        dones = np.array(data['dones'], dtype=np.float32)
                        useful = np.array(data['useful'], dtype=bool)
                        weights = np.array(data['weights'], dtype=np.float32)
                        log_prob = np.array(data['log_prob'], dtype=np.float32)

                        # compute discounted returns per episode
                        returns = np.zeros_like(rews, dtype=np.float32)
                        G = 0.0
                        for i in reversed(range(len(rews))):
                            if dones[i]:
                                G = rews[i]
                            else:
                                G = rews[i] + gamma * G
                            returns[i] = G

                        unified_weights = np.zeros_like(weights, dtype=np.float32)
                        cur_weight = 1.0
                        idx_start = 0
                        for i in range(1, len(weights)):
                            if weights[i] > 0 and weights[i] != weights[i-1]:
                                cur_weight *= weights[i]
                            if dones[i]:
                                # cur_weight = max(cur_weight, 0.1)
                                unified_weights[idx_start:i+1] = cur_weight
                                idx_start = i + 1
                                cur_weight = 1.0

                        useful_idx = np.where(useful)[0]
                        if len(useful_idx) == 0:
                            continue

                        tmp_obs = np.concatenate([tmp_obs, obs[useful_idx]])
                        tmp_acts = np.concatenate([tmp_acts, acts[useful_idx]])
                        tmp_returns = np.concatenate([tmp_returns, returns[useful_idx]])
                        tmp_weights = np.concatenate([tmp_weights, unified_weights[useful_idx]])
                        tmp_log_prob = np.concatenate([tmp_log_prob, log_prob[useful_idx]])
                
                np.save(all_data_path, {'obs': tmp_obs, 'actions': tmp_acts, 'returns': tmp_returns, 'weights': tmp_weights, 'log_prob': tmp_log_prob})
                all_obs = np.concatenate([all_obs, tmp_obs])
                all_acts = np.concatenate([all_acts, tmp_acts])
                all_returns = np.concatenate([all_returns, tmp_returns])
                all_weights = np.concatenate([all_weights, tmp_weights])
                all_log_prob = np.concatenate([all_log_prob, tmp_log_prob])

        obs_t = torch.tensor(all_obs, dtype=torch.float32)
        acts_t = torch.tensor(all_acts, dtype=torch.float32)
        returns_t = torch.tensor(all_returns, dtype=torch.float32)
        weights_t = torch.tensor(all_weights, dtype=torch.float32)
        weights_t = weights_t.clamp(max=1e-1)
        weights_t = weights_t / (weights_t.mean() + 1e-8)
        weights_t = torch.clamp(weights_t, max=30.0)
        log_prob_t = torch.tensor(all_log_prob, dtype=torch.float32)
        print(f'Loaded offline dataset with {len(obs_t)} samples from {len(data_dirs)} directories')

        if args.debug_first_batch:
            def _stats(name, t):
                q = torch.quantile(t, torch.tensor([0.0, 0.5, 0.9, 0.99, 1.0])).tolist()
                print(f'  {name:10s} shape={tuple(t.shape)} '
                      f'min={q[0]:.4f} p50={q[1]:.4f} p90={q[2]:.4f} p99={q[3]:.4f} max={q[4]:.4f} '
                      f'mean={t.mean().item():.4f} std={t.std().item():.4f}')
            print('[dataset stats]')
            _stats('returns', returns_t)
            _stats('weights', weights_t)
            _stats('log_prob', log_prob_t)
            # per-dim action range (helps judge if log_std is reasonable)
            print(f'  actions   min={acts_t.min().item():.4f} max={acts_t.max().item():.4f} '
                  f'per-dim std={acts_t.std(dim=0).mean().item():.4f}')

        return obs_t, acts_t, returns_t, weights_t, log_prob_t

    def offline_train_policy(model, dataset_path, epochs, batch_size, lr, value_coef, device='cpu'):
        from torch.utils.data import TensorDataset, DataLoader, random_split

        obs_t, acts_t, returns_t, weights_t, log_prob_t = load_offline_dataset(dataset_path)
        ds = TensorDataset(obs_t, acts_t, returns_t, weights_t, log_prob_t)

        # Train/val split: used to (a) select best checkpoint on val, (b) drive early stopping.
        n_total = len(ds)
        n_val = int(n_total * args.val_split) if args.val_split > 0 else 0
        n_train = n_total - n_val
        from torch.utils.data import WeightedRandomSampler

        def _make_train_loader(train_ds_):
            """Build training dataloader. If weighted sampling is enabled, sample with
            replacement proportional to b_weights so each batch over-represents rare
            high-weight samples. When sampler is used, shuffle is disabled."""
            if args.use_weighted_sampler:
                # train_ds_ may be a Subset; pull the underlying weights tensor for its indices
                if hasattr(train_ds_, 'indices'):
                    w_tensor = ds.tensors[3][train_ds_.indices]
                else:
                    w_tensor = ds.tensors[3]
                w_sample = w_tensor.clamp(min=1e-4).double()
                sampler = WeightedRandomSampler(
                    weights=w_sample, num_samples=len(train_ds_), replacement=True
                )
                return DataLoader(train_ds_, batch_size=batch_size, sampler=sampler)
            else:
                return DataLoader(train_ds_, batch_size=batch_size, shuffle=True)

        if n_val > 0:
            train_ds, val_ds = random_split(
                ds, [n_train, n_val],
                generator=torch.Generator().manual_seed(42),
            )
            train_dl = _make_train_loader(train_ds)
            val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        else:
            train_dl = _make_train_loader(ds)
            val_dl = None
        if args.use_weighted_sampler:
            print('[Offline] using WeightedRandomSampler; in-loss b_weights reweighting disabled.')
        print(f'[Offline] train samples={n_train}, val samples={n_val}')

        policy = model.policy
        params = [p for p in policy.parameters() if p.requires_grad]
        optim = torch.optim.Adam(params, lr=lr)
        if args.initial.endswith('.pt') and args.use_initial_optimizer:
            if "optimizer_state_dict" in state_dict.keys():
                optim.load_state_dict(state_dict["optimizer_state_dict"])
                print(f'Loaded optimizer state from {args.initial}')

        if args.train_value_net_only:
            for name, param in policy.named_parameters():
                if 'value' not in name:
                    param.requires_grad = False
            print('Training value net only, policy net weights frozen.')

        _debug_state = {'printed': False}

        # Frozen snapshot of the original (pretrained) policy for action-drift diagnostic.
        # Copy the current policy's state_dict right after pretrain loading so it captures
        # the init-time actor weights, then freeze.
        import copy as _copy
        orig_policy_snapshot = _copy.deepcopy(policy).to(device).eval()
        for p in orig_policy_snapshot.parameters():
            p.requires_grad = False

        def action_diff_on_loader(loader):
            """Return (mse, mean_abs, max_abs, per_dim_std_ratio) between new and orig policy
            predicted actions, computed over `loader`. Averaged over samples."""
            if loader is None:
                return None
            policy.eval()
            sq_sum = 0.0
            abs_sum = 0.0
            max_abs = 0.0
            n_tot = 0
            new_acts_chunks = []
            orig_acts_chunks = []
            with torch.no_grad():
                for batch in loader:
                    b_obs = batch[0].to(device)
                    # new policy pred
                    lat_new = policy.mlp_extractor.policy_net(b_obs)
                    a_new = policy.action_net(lat_new)
                    # orig policy pred (frozen)
                    lat_o = orig_policy_snapshot.mlp_extractor.policy_net(b_obs)
                    a_orig = orig_policy_snapshot.action_net(lat_o)
                    d = a_new - a_orig
                    sq_sum += float((d * d).sum().item())
                    abs_sum += float(d.abs().sum().item())
                    max_abs = max(max_abs, float(d.abs().max().item()))
                    n_tot += b_obs.shape[0] * a_new.shape[1]
                    new_acts_chunks.append(a_new.detach().cpu())
                    orig_acts_chunks.append(a_orig.detach().cpu())
            mse = sq_sum / max(1, n_tot)
            mae = abs_sum / max(1, n_tot)
            # action std ratio (per-dim): indicates whether new policy collapsed or expanded vs orig
            new_all = torch.cat(new_acts_chunks, dim=0)
            orig_all = torch.cat(orig_acts_chunks, dim=0)
            std_ratio = float((new_all.std(dim=0) / (orig_all.std(dim=0) + 1e-8)).mean().item())
            return mse, mae, max_abs, std_ratio

        def compute_batch_loss(batch, in_warmup: bool):
            b_obs, b_act, b_ret, b_weights, b_log_prob = [t.to(device) for t in batch]

            # Forward
            latent_pi = policy.mlp_extractor.policy_net(b_obs)
            pred_act = policy.action_net(latent_pi)
            latent_v = policy.mlp_extractor.value_net(b_obs)
            pred_val = policy.value_net(latent_v).squeeze(-1)

            # When using WeightedRandomSampler, samples are already drawn proportional to
            # b_weights, so the per-sample loss weight becomes 1 (double-counting otherwise).
            if args.use_weighted_sampler:
                loss_w = torch.ones_like(b_weights)
            else:
                loss_w = b_weights

            # Value loss: Huber/MSE, robust to outliers
            if args.value_loss == 'huber':
                val_per_sample = F.smooth_l1_loss(pred_val, b_ret, reduction='none', beta=args.huber_delta)
            else:
                val_per_sample = F.mse_loss(pred_val, b_ret, reduction='none')
            val_loss = (val_per_sample * loss_w).mean()

            # log-prob under current policy with trainable per-dim log_std.
            log_std = torch.clamp(policy.log_std, min=args.log_std_min, max=args.log_std_max)
            log_std = log_std.expand_as(pred_act)
            std = torch.exp(log_std)
            dist = torch.distributions.Normal(pred_act, std)
            logp = dist.log_prob(b_act).sum(dim=1)

            bc_per_sample = ((pred_act - b_act) ** 2).sum(dim=1)

            if args.filter_bc:
                # Filter-BC: imitate only high-return (non-crash) samples.
                keep = (b_ret > args.filter_return_threshold).float()

                if args.filter_awac:
                    # Hybrid: inside kept samples, weight by exp(adv/beta) so better-than-V
                    # non-crash samples pull harder. Crash samples remain zeroed by keep=0.
                    adv = b_ret - pred_val.detach()
                    # Normalize using only kept samples so the scale is meaningful.
                    if keep.sum() > 1:
                        kept_adv = adv[keep.bool()]
                        adv_norm = (adv - kept_adv.mean()) / (kept_adv.std() + 1e-8)
                    else:
                        adv_norm = adv
                    awac_w = torch.exp(adv_norm / args.awac_beta).clamp(max=args.awac_weight_max)
                else:
                    awac_w = torch.ones_like(b_ret)

                w_keep = (awac_w * loss_w * keep).clamp(max=args.combined_weight_max)
                denom = w_keep.sum().clamp(min=1.0)

                policy_loss = -(w_keep * logp).sum() / denom
                bc_loss = (w_keep * bc_per_sample).sum() / denom

                # Hinge repulsion on crash samples: push policy mean away from crash actions in
                # action space. relu(margin - dist^2) is zero once the policy is far enough,
                # preventing the unbounded divergence that logp-based repulsion causes.
                if args.crash_coef > 0:
                    crash_mask = 1.0 - keep
                    dist_sq = ((pred_act - b_act) ** 2).sum(dim=1)
                    crash_repel = F.relu(args.crash_margin - dist_sq)
                    return_weight = (-b_ret).clamp(0.0, 1.0)  # |return|: 1.0 at crash step, ~0.5 at threshold boundary
                    w_crash = (loss_w * crash_mask * return_weight).clamp(max=args.combined_weight_max)
                    crash_denom = w_crash.sum().clamp(min=1.0)
                    crash_loss = (w_crash * crash_repel).sum() / crash_denom
                else:
                    crash_loss = torch.zeros((), device=pred_act.device)

                combined_w = w_keep
            else:
                crash_loss = torch.zeros((), device=pred_act.device)
                # Per-batch normalized advantage for stable AWAC weights
                adv = b_ret - pred_val.detach()
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                # AWAC / AWR exponential advantage weights (no importance ratio)
                awac_w = torch.exp(adv / args.awac_beta).clamp(max=args.awac_weight_max)

                # Combine AWAC weight with loss_w, then cap the combined product so
                # no single sample dominates the batch gradient.
                combined_w = (awac_w * loss_w).clamp(max=args.combined_weight_max)

                # AWAC/AWR policy loss
                policy_loss = -(combined_w * logp).mean()

                # KL-to-behavior under fixed std ~ scaled action MSE; trust-region regularizer
                bc_loss = (bc_per_sample * loss_w.clamp(max=args.combined_weight_max)).mean()

            # Out-of-bounds soft penalty: discourages pred_act from extrapolating outside the
            # data range on OOD states. Zero inside [-bound, +bound], quadratic outside.
            if args.oob_coef > 0:
                over = (pred_act.abs() - args.oob_bound).clamp(min=0.0)
                oob_loss = (over * over).sum(dim=1).mean()
            else:
                oob_loss = torch.zeros((), device=pred_act.device)

            if args.train_value_net_only or in_warmup:
                total = val_loss
            else:
                total = (policy_loss
                         + value_coef * val_loss
                         + args.bc_coef * bc_loss
                         + args.oob_coef * oob_loss
                         + args.crash_coef * crash_loss)

            # One-shot diagnostic on the very first batch
            if args.debug_first_batch and not _debug_state['printed']:
                _debug_state['printed'] = True
                with torch.no_grad():
                    print('[first-batch debug]')
                    print(f'  pred_val  min={pred_val.min().item():.3f} max={pred_val.max().item():.3f} '
                          f'mean={pred_val.mean().item():.3f}')
                    print(f'  b_ret     min={b_ret.min().item():.3f} max={b_ret.max().item():.3f} '
                          f'mean={b_ret.mean().item():.3f}')
                    print(f'  b_weights min={b_weights.min().item():.3f} max={b_weights.max().item():.3f} '
                          f'mean={b_weights.mean().item():.3f}')
                    print(f'  awac_w    min={awac_w.min().item():.3f} max={awac_w.max().item():.3f} '
                          f'mean={awac_w.mean().item():.3f}')
                    print(f'  combined  min={combined_w.min().item():.3f} max={combined_w.max().item():.3f} '
                          f'mean={combined_w.mean().item():.3f}')
                    print(f'  val/s     min={val_per_sample.min().item():.3f} max={val_per_sample.max().item():.3f} '
                          f'mean={val_per_sample.mean().item():.3f}')
                    print(f'  logp      min={logp.min().item():.2f} max={logp.max().item():.2f} '
                          f'mean={logp.mean().item():.2f}')

            return total, policy_loss.detach(), val_loss.detach(), bc_loss.detach(), oob_loss.detach(), crash_loss.detach()

        min_val_metric = float('inf')
        min_train_metric = float('inf')
        patience = 0

        for ep in range(1, epochs + 1):
            in_warmup = (not args.train_value_net_only) and (ep <= args.value_warmup_epochs)

            # ---- Train ----
            policy.train()
            ep_total = ep_pi = ep_v = ep_bc = ep_oob = ep_crash = 0.0
            n_batch = 0
            for batch in train_dl:
                total, pi_l, v_l, bc_l, oob_l, crash_l = compute_batch_loss(batch, in_warmup)
                optim.zero_grad()
                total.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=args.max_grad_norm)
                optim.step()

                ep_total += float(total.item())
                ep_pi += float(pi_l.item())
                ep_v += float(v_l.item())
                ep_bc += float(bc_l.item())
                ep_oob += float(oob_l.item())
                ep_crash += float(crash_l.item())
                n_batch += 1

            train_avg = ep_total / max(1, n_batch)
            pi_avg = ep_pi / max(1, n_batch)
            v_avg = ep_v / max(1, n_batch)
            bc_avg = ep_bc / max(1, n_batch)
            oob_avg = ep_oob / max(1, n_batch)
            crash_avg = ep_crash / max(1, n_batch)

            # ---- Validation ----
            val_avg = None
            if val_dl is not None:
                policy.eval()
                v_sum, v_n = 0.0, 0
                with torch.no_grad():
                    for batch in val_dl:
                        total, _, _, _, _, _ = compute_batch_loss(batch, in_warmup)
                        v_sum += float(total.item())
                        v_n += 1
                val_avg = v_sum / max(1, v_n)

            with torch.no_grad():
                ls = policy.log_std.detach().cpu().numpy()
            ls_summary = f'log_std[min/mean/max]={ls.min():.3f}/{ls.mean():.3f}/{ls.max():.3f}'

            # Action drift vs original (pretrained) policy, on val set if available else train.
            diff_loader = val_dl if val_dl is not None else train_dl
            drift = action_diff_on_loader(diff_loader)
            if drift is not None:
                d_mse, d_mae, d_maxabs, d_stdratio = drift
                drift_summary = (f'drift[mse={d_mse:.4f} mae={d_mae:.4f} '
                                 f'max={d_maxabs:.3f} std_ratio={d_stdratio:.3f}]')
            else:
                drift_summary = ''

            tag = '[warmup]' if in_warmup else '[train] '
            if val_avg is not None:
                print(f'[Offline]{tag} ep={ep}/{epochs} train={train_avg:.6f} val={val_avg:.6f} '
                      f'pi={pi_avg:.6f} v={v_avg:.6f} bc={bc_avg:.6f} oob={oob_avg:.6f} crash={crash_avg:.6f} {ls_summary} {drift_summary}')
            else:
                print(f'[Offline]{tag} ep={ep}/{epochs} train={train_avg:.6f} '
                      f'pi={pi_avg:.6f} v={v_avg:.6f} bc={bc_avg:.6f} oob={oob_avg:.6f} crash={crash_avg:.6f} {ls_summary} {drift_summary}')

            # ---- Best checkpoint selection (joint-training epochs only) ----
            # Skip during warmup: warmup loss is pure value loss, not comparable to the joint
            # loss and not what we ultimately want to select on.
            if not in_warmup:
                sel_metric = val_avg if val_avg is not None else train_avg
                if val_avg is not None:
                    improved = sel_metric < min_val_metric
                    if improved:
                        min_val_metric = sel_metric
                else:
                    improved = sel_metric < min_train_metric
                    if improved:
                        min_train_metric = sel_metric

                if improved:
                    save_to = os.path.join(args.out, 'best.policy.pt')
                    safe_model_save(model, save_to, verbose=0, optimizer=optim)
                    src = 'val' if val_avg is not None else 'train'
                    print(f'[Offline] new best @ ep {ep} ({src}={sel_metric:.6f})')
                    patience = 0
                else:
                    patience += 1
                    if args.early_stop_patience > 0 and patience >= args.early_stop_patience:
                        print(f'[Offline] early stop at ep {ep} (no improvement for {patience} epochs)')
                        break

            # Per-epoch checkpoint (kept on user request)
            try:
                save_to = os.path.join(args.out, f'ep{ep}.policy.pt')
                safe_model_save(model, save_to, verbose=0, optimizer=optim)
            except Exception as e:
                print('[Offline] saving failed:', e)


    def evaluate_policy(safe_policy, policy=pretrain_wrapper, criticality_model=criticality_model, n_episodes: int = 1):
        config_file_path = "go2_training.yaml"
        terrain_cfg = "terrain_config.yaml"

        policy_net = safe_policy.mlp_extractor.policy_net
        action_net = safe_policy.action_net
        policy_net.to('cpu')
        action_net.to('cpu')
        policy_net.eval()
        action_net.eval()

        safe_wrapper = PolicyOnlyWrapper(policy_net, action_net).cpu()

        trainer = TestEnv(policy=policy, safe_policy=safe_wrapper, criticality_model=criticality_model, config_file_path=config_file_path, terrain_config_file=terrain_cfg, critical_threshold=0.5)
        env = TerrainGymEnv(trainer, max_episode_steps=args.max_steps)

        action_space = env.action_space
        # prepare discretization edges for actions: 10 bins per dimension
        low = np.asarray(action_space.low, dtype=np.float32)
        high = np.asarray(action_space.high, dtype=np.float32)
        high = np.where(high == low, low + 1.0, high)
        action_edges = [np.linspace(low[d], high[d], num=11) for d in range(low.shape[0])]

        results = []
        for i in range(n_episodes):
            obs, _ = env.reset()
            done = False

            # per-episode accumulators
            ep_steps = 0
            ep_total_reward = 0.0

            while not done:
                ep_steps += 1
                bins = np.random.randint(0, 10, size=action_space.shape)
                centers = np.zeros(action_space.shape, dtype=np.float32)
                flat_bins = np.asarray(bins).reshape(-1)
                for d in range(flat_bins.shape[0]):
                    b = int(flat_bins[d])
                    e = action_edges[d]
                    centers.reshape(-1)[d] = 0.5 * (e[b] + e[b+1])
                action = centers
                action = np.asarray(action, dtype=np.float32)

                next_obs, reward, terminated, truncated, info = env.step(action)
                obs = next_obs

                ep_total_reward += reward

                done = bool(terminated) or bool(truncated) or bool(info.get('fallen', False) or info.get('collided', False) or info.get('base_collision', False) or info.get('thigh_collision', False) or info.get('stuck', False))

            # episode finished; determine crash/failure from last step's info
            crash = int(bool(info.get('fallen', False) or info.get('collided', False) or info.get('base_collision', False) or info.get('thigh_collision', False) or info.get('stuck', False)))
            crash_type = 'fallen' if info.get('fallen', False) else 'collided' if info.get('collided', False) else 'base_collision' if info.get('base_collision', False) else 'thigh_collision' if info.get('thigh_collision', False) else 'stuck' if info.get('stuck', False) else 'safe'

            # print per-episode breakdown and total
            print(f"episode {i+1}/{n_episodes} steps={ep_steps} crash={crash}, type={crash_type}")
            print(f"  total_reward (accumulated) = {ep_total_reward:.6f}")
            results.append(ep_total_reward)

        return results

    def safe_model_save(model_obj, save_path, verbose=1, optimizer=None):
        """Save `model_obj` while temporarily clearing attributes that may be
        unpickleable (eg. multiprocessing auth keys inside VecEnv/processes).
        Restores cleared attributes after the save attempt.
        """
        cleared = {}
        for attr in ('env', 'envs', 'training_env', 'vec_env'):
            if hasattr(model_obj, attr):
                cleared[attr] = getattr(model_obj, attr)
                try:
                    setattr(model_obj, attr, None)
                except Exception:
                    # best-effort: if we can't clear it, ignore and continue
                    cleared.pop(attr, None)

        try:
            # try to save model policy state dict and optional optimizer state dict with torch
            fb_dir = os.path.dirname(save_path)
            if fb_dir:
                os.makedirs(fb_dir, exist_ok=True)
            fb_path = save_path
            # prefer .pt for fallback
            if not fb_path.endswith('.pt'):
                fb_path = save_path + '.pt'
            payload = {'policy_state_dict': model_obj.policy.state_dict()}
            if optimizer is not None:
                try:
                    payload['optimizer_state_dict'] = optimizer.state_dict()
                except Exception:
                    # best-effort: ignore optimizer if it can't be serialized
                    if verbose:
                        print('[safe_model_save] warning: failed to include optimizer state_dict')
            torch.save(payload, fb_path)
            if verbose:
                print(f'[safe_model_save] saved policy (and optimizer) to {fb_path}')
            return True
        finally:
            for attr, val in cleared.items():
                try:
                    setattr(model_obj, attr, val)
                except Exception:
                    pass

    print('Starting offline PPO-style training (adv-weighted BC + value regression)')
    offline_train_policy(model, args.dataset, args.offline_epochs, args.offline_batch_size, args.offline_lr, args.value_coef, device=args.rl_device)
    # evaluate after offline training
    _ = evaluate_policy(model.policy, n_episodes=int(args.n_eval_episodes))

    save_path = os.path.join(args.out, f'{args.run_name}_ppo.zip')
    if safe_model_save(model, save_path, verbose=1):
        print('Saved model to', save_path)
    else:
        print('Failed to save final model to', save_path)

if __name__ == '__main__':
    main()
