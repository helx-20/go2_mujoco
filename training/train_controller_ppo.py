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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import yaml
import numpy as np
from stable_baselines3 import PPO
import types
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.buffers import RolloutBuffer
from training.utils.train_env import TrainEnv
from training.utils.test_env import TestEnv, TerrainGymEnv

# Silence Gym -> Gymnasium migration warning and noisy gym/gymnasium loggers
import warnings
import logging
warnings.filterwarnings("ignore", message=".*Please upgrade to Gymnasium.*", category=UserWarning)
logging.getLogger("gym").setLevel(logging.ERROR)
logging.getLogger("gymnasium").setLevel(logging.ERROR)


def make_env_fn(warmup_policy, max_episode_steps=1000, nade=False, nade_model_path=None):
    def _thunk():
        # Create a lightweight dummy policy to satisfy TestEnv/Go2Controller
        config_file_path = "go2_training.yaml"
        terrain_cfg = "terrain_config.yaml"
        trainer = TestEnv(policy=warmup_policy, config_file_path=config_file_path, terrain_config_file=terrain_cfg)
        env = TrainEnv(trainer=trainer, max_episode_steps=max_episode_steps, nade=nade, nade_model_path=nade_model_path)
        return Monitor(env)

    return _thunk

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--total_timesteps', type=int, default=3000000)
    parser.add_argument('--num_envs', type=int, default=16)
    parser.add_argument('--sim_device', type=str, default='cpu')
    parser.add_argument('--rl_device', type=str, default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_steps', type=int, default=30)
    parser.add_argument('--log_interval', type=int, default=20000, help='Timesteps between simple stdout progress prints')
    parser.add_argument('--eval_freq', type=int, default=20000, help='Timesteps between evaluations')
    parser.add_argument('--n_eval_episodes', type=int, default=5, help='Number of episodes per evaluation')
    parser.add_argument('--tensorboard_log', type=str, default=None, help='TensorBoard log dir')
    parser.add_argument('--out', type=str, default='training/models/')
    parser.add_argument('--run_name', type=str, default='run1')
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--ent_coef', type=float, default=0.01)
    parser.add_argument('--n_steps', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--pretrain', type=str, default='training/models/actor_init.zip',
                        help='Path to a pretrained PyTorch model or SB3 .zip to initialize policy (default uses training/models/actor_init.zip)')
    parser.add_argument('--nade', action='store_true', help='Whether to use NADE policy architecture (default: False)')
    parser.add_argument('--nade_model_path', type=str, default='criticality/stage1_plus/model/stage1_plus_criticality_best_new_3.pt', help='Path to NADE policy model (used if --nade is set)')
    args = parser.parse_args()
    args.out = os.path.join(args.out, args.run_name)
    if not args.tensorboard_log:
        args.tensorboard_log = os.path.join(args.out, 'logs')
    print(args)

    os.makedirs(args.out, exist_ok=True)

    # If a pretrained SB3 .zip is provided, load it only for weight extraction (no env attachment)
    # We will always create a fresh PPO model below and copy weights into it to avoid carrying over
    # stale internal training state (rollout buffer, optim state, etc.) from the checkpoint.
    sb3_pretrain_model = None
    if args.pretrain and os.path.isfile(args.pretrain) and args.pretrain.endswith('.zip'):
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
    wrapper = PolicyOnlyWrapper(sb3_pretrain_model.policy.mlp_extractor.policy_net.to('cpu').eval(), sb3_pretrain_model.policy.action_net.to('cpu').eval()).cpu()
    
    env_fns = [make_env_fn(wrapper, args.max_steps, args.nade, args.nade_model_path) for _ in range(args.num_envs)]
    if args.num_envs == 1:
        vec_env = DummyVecEnv(env_fns)
    else:
        vec_env = SubprocVecEnv(env_fns)

    # Create a fresh model and try to copy compatible weights if available
    hidden_sizes = [512, 256, 128]
    policy_kwargs = {'net_arch': dict(pi=hidden_sizes, vf=hidden_sizes), 'activation_fn': torch.nn.ELU}
    # Create model with CLI-configured hyperparameters
    model = PPO(
        'MlpPolicy', vec_env, verbose=1, device=args.rl_device, seed=args.seed,
        policy_kwargs=policy_kwargs,
        learning_rate=float(args.learning_rate),
        ent_coef=float(args.ent_coef),
        n_steps=int(args.n_steps),
        batch_size=int(args.batch_size),
    )

    if args.pretrain and os.path.isfile(args.pretrain):
        # prefer already-loaded sb3_pretrain_model if available
        sb3_model = sb3_pretrain_model if sb3_pretrain_model is not None else __import__('stable_baselines3').PPO.load(args.pretrain, device='cpu')

        # Source and destination state dicts
        src_sd = sb3_model.policy.state_dict()
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
        print(f'Initialized policy from {args.pretrain} — matched {len(matched)} tensors')

    # Override `model.predict` to use the raw PolicyOnlyWrapper forward pass
    try:
        def _predict_using_wrapper(self, observation, deterministic=True):
            out = model.policy.mlp_extractor.policy_net(observation)
            out = model.policy.action_net(out)
            return out, None

        model.predict = types.MethodType(_predict_using_wrapper, model)
        print('Patched model.predict to use PolicyOnlyWrapper (raw network outputs)')
    except Exception as e:
        print('Failed to patch model.predict:', e)

    # Print chosen hyperparameters and ensure model prints some progress; set verbose and attach a simple callback
    # Ensure model.n_steps/batch_size match args and recreate rollout_buffer to expected total size
    # enforce model attributes
    model.n_steps = int(args.n_steps)
    model.batch_size = int(args.batch_size)

    # RolloutBuffer buffer_size should be number of steps (n_steps),
    # observations are stored as (buffer_size, n_envs, ...)
    expected_size = int(args.n_steps)
    rb = getattr(model, 'rollout_buffer', None)
    current_size = getattr(rb, 'buffer_size', None) if rb is not None else None
    if current_size != expected_size:
        print(f'Forcing rollout_buffer.buffer_size -> {expected_size} (was {current_size})')
        obs_space = vec_env.observation_space
        act_space = vec_env.action_space
        device = getattr(model, 'device', args.rl_device)
        gamma = getattr(model, 'gamma', 0.99)
        gae_lambda = getattr(model, 'gae_lambda', 0.95)
        model.rollout_buffer = RolloutBuffer(buffer_size=expected_size,
                                            observation_space=obs_space,
                                            action_space=act_space,
                                            device=device,
                                            gamma=gamma,
                                            gae_lambda=gae_lambda,
                                            n_envs=int(args.num_envs))

    print('Training with hyperparameters:', dict(learning_rate=args.learning_rate, ent_coef=args.ent_coef, n_steps=args.n_steps, batch_size=args.batch_size))
    model.verbose = max(getattr(model, 'verbose', 0), 1)

    # configure logger (tensorboard/csv/stdout)
    tb_dir = os.path.join(args.tensorboard_log, args.run_name)
    new_logger = configure(tb_dir, ["stdout", "tensorboard", "csv"])
    model.set_logger(new_logger)

    def evaluate_policy(policy, n_episodes: int = 5):
        config_file_path = "go2_training.yaml"
        terrain_cfg = "terrain_config.yaml"

        policy_net = policy.mlp_extractor.policy_net
        action_net = policy.action_net
        policy_net.to('cpu')
        action_net.to('cpu')
        policy_net.eval()
        action_net.eval()

        wrapper = PolicyOnlyWrapper(policy_net, action_net).cpu()

        trainer = TestEnv(policy=wrapper, config_file_path=config_file_path, terrain_config_file=terrain_cfg)
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

    def safe_model_save(model_obj, save_path, verbose=1):
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
            model_obj.save(save_path)
            if verbose:
                print(f'[safe_model_save] Saved model to {save_path}')
            return True
        except Exception as e:
            if verbose:
                print(f'[safe_model_save] failed to save model: {e}')
            # Fallback: try to save only the policy state dict with torch
            try:
                fb_dir = os.path.dirname(save_path)
                if fb_dir:
                    os.makedirs(fb_dir, exist_ok=True)
                fb_path = save_path
                # prefer .pt for fallback
                if not fb_path.endswith('.pt'):
                    fb_path = save_path + '.policy.pt'
                torch.save({'policy_state_dict': model_obj.policy.state_dict()}, fb_path)
                if verbose:
                    print(f'[safe_model_save] Fallback: saved policy state_dict to {fb_path}')
                return True
            except Exception as e2:
                if verbose:
                    print(f'[safe_model_save] fallback save also failed: {e2}')
                return False
        finally:
            for attr, val in cleared.items():
                try:
                    setattr(model_obj, attr, val)
                except Exception:
                    pass

    # _ = evaluate_policy(model.policy, n_episodes=2)

    class PolicyNetEvalCallback(BaseCallback):
        """Eval callback that runs episodes using `evaluate_policy(policy, ...)`.
        Saves best model to `best_model_save_path` when mean reward improves.
        """
        def __init__(self, best_model_save_path, n_eval_episodes=5, eval_freq=10000, verbose=1):
            super().__init__(verbose)
            self.best_model_save_path = best_model_save_path
            self.n_eval_episodes = int(n_eval_episodes)
            self.eval_freq = int(eval_freq)
            self.best_mean_reward = -float('inf')
            self._last_eval_ts = 0

        def _on_step(self) -> bool:
            if (self.num_timesteps - self._last_eval_ts) < self.eval_freq:
                return True
            self._last_eval_ts = self.num_timesteps

            # use evaluate_policy to compute per-episode totals
            try:
                eval_results = evaluate_policy(self.model.policy, n_episodes=self.n_eval_episodes)
            except Exception as e:
                if self.verbose:
                    print(f'[PolicyNetEval] evaluation failed: {e}')
                return True

            results = eval_results if isinstance(eval_results, (list, tuple)) else [float(eval_results)]

            if len(results) > 0:
                mean_r = float(sum(results) / len(results))
            else:
                mean_r = -float('inf')

            if self.verbose:
                print(f'[PolicyNetEval] timesteps={self.num_timesteps} mean_reward={mean_r:.6f}')

            # save best model (same behavior as original)
            if mean_r >= self.best_mean_reward:
                self.best_mean_reward = mean_r
                os.makedirs(self.best_model_save_path, exist_ok=True)
                save_to = os.path.join(self.best_model_save_path, 'best_model.zip')
                try:
                    ok = safe_model_save(self.model, save_to, verbose=self.verbose)
                    if ok and self.verbose:
                        print(f'[PolicyNetEval] New best mean_reward={mean_r:.6f}, saved to {save_to}')
                    elif not ok and self.verbose:
                        print(f'[PolicyNetEval] failed to save model to {save_to}')
                except Exception as e:
                    if self.verbose:
                        print(f'[PolicyNetEval] failed to save model: {e}')
            
            safe_model_save(self.model, os.path.join(self.best_model_save_path, f'model_ep{self.num_timesteps}.zip'), verbose=self.verbose)

            return True

    eval_callback = PolicyNetEvalCallback(best_model_save_path=os.path.join(args.out, 'best'),
                                         n_eval_episodes=int(args.n_eval_episodes), eval_freq=int(args.eval_freq),
                                         verbose=1)

    class ProgressPrintCallback(BaseCallback):
        """Simple callback that prints progress every `log_interval` timesteps."""
        def __init__(self, log_interval: int = 1000, verbose=0):
            super().__init__(verbose)
            self.log_interval = int(log_interval)

        def _on_step(self) -> bool:
            if self.num_timesteps % self.log_interval == 0:
                print(f'[SB3] timesteps={self.num_timesteps}')

            return True

    pb_cb = ProgressPrintCallback(log_interval=args.log_interval)

    model.learn(total_timesteps=int(args.total_timesteps), callback=[pb_cb, eval_callback])

    save_path = os.path.join(args.out, f'{args.run_name}_ppo.zip')
    if safe_model_save(model, save_path, verbose=1):
        print('Saved model to', save_path)
    else:
        print('Failed to save final model to', save_path)

if __name__ == '__main__':
    main()
