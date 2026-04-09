import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import gymnasium as gym
import gym_testenvs
from stable_baselines3 import DQN,PPO
from stable_baselines3.common.evaluation import evaluate_policy

import matplotlib.pyplot as plt
import torch
import argparse 

from scipy.stats import norm
from criticality.utils.criticality_model import SimpleClassifier
# NOTE: original code used Reward_Model from Env_agent; we adapt by loading a SimpleClassifier

import numpy as np
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

def convert_to_serializable(obj):
    if isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    else:
        return obj

def generate_seed():
    seed = os.getpid() + int(time.time() * 1e5 % 1e6)
    return seed

def main(args):
    device = torch.device("cuda")
    max_epoch = 100000
    model = PPO.load("Rocket_agent_withwind/model/ppo_lunar.pkl.zip",device='cpu')

    def build_wrapper_from_state(path, device=torch.device('cpu')):
        if not path or not os.path.exists(path):
            return None
        state = torch.load(path, map_location=torch.device('cpu'))
        input_dim = None
        if isinstance(state, dict):
            for key in state.keys():
                if key.endswith('net.0.weight'):
                    try:
                        input_dim = state[key].shape[1]
                        break
                    except Exception:
                        pass
        if input_dim is None:
            return None
        model = SimpleClassifier(input_dim=input_dim, hidden=256)
        try:
            model.load_state_dict(state)
        except Exception:
            pass
        model.eval().to(device)
        return model

    pre_model1 = build_wrapper_from_state(args.model_path, device=torch.device('cuda'))
    if pre_model1 is None:
        print(f'Warning: failed to build criticality model from {args.model_path}; proceeding without it')
    
    WIND_POWER = 10.0
    TURBULENCE_POWER = 1.0
    env = gym.make('LunarLander/ordinary_nade-v0', gravity=-8.5, enable_wind=True, wind_power = WIND_POWER, turbulence_power = TURBULENCE_POWER, criticality_model1 = pre_model1, device=device, criticality_thresh=args.criticality_thresh, epsilon=args.epsilon, min_weight=args.min_weight)

    start_epoch = 1
    
    bins_per_dim = getattr(args, 'bins_per_dim', 10)
    D = 4
    edges = np.linspace(-1.0, 1.0, bins_per_dim + 1)
    grids = np.meshgrid(*[np.arange(bins_per_dim) for _ in range(D)], indexing='ij')
    bins_flat = np.stack([g.reshape(-1) for g in grids], axis=1).astype(np.int64)
    num_actions = bins_flat.shape[0]
    centers = np.zeros((num_actions, D), dtype=np.float32)
    for d in range(D):
        b_idx = bins_flat[:, d]
        centers[:, d] = 0.5 * (edges[b_idx] + edges[b_idx + 1])

    for epoch in range(start_epoch, max_epoch):  
        obs,_= env.reset()
        done = False
        
        episode_data = {}
        
        weight_step_info = {}
        drl_epsilon_step_info = {}
        ndd_step_info = {}
        drl_obs_step_info = {}
        
        weight = 1
        control_step = 0
        i = 0
        steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            # ensure action is numpy array of length >=4
            try:
                if hasattr(action, 'cpu'):
                    action = action.cpu().numpy()
            except Exception:
                pass
            action = np.asarray(action, dtype=np.float32).reshape(-1)

            # evaluate criticality model on obs+action candidates
            if pre_model1 is None:
                p_list = np.ones(num_actions, dtype=float) / float(num_actions)
                pdf_array = p_list
                action_idx = int(np.argmin(np.linalg.norm(centers - (action[:D].reshape(1, -1)), axis=1)))
                cur_weight = 1.0
            else:
                with torch.no_grad():
                    t_obs = torch.from_numpy(np.asarray(obs, dtype=np.float32)).to(device).unsqueeze(0).repeat(num_actions, 1)
                    t_act = torch.from_numpy(centers).to(device)
                    t_in = torch.cat([t_obs, t_act], dim=1)
                    out = pre_model1(t_in)
                    if isinstance(out, torch.Tensor) and out.dim() == 2 and out.size(1) == 2:
                        scores = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
                    else:
                        scores = out.view(-1).cpu().numpy()
                # construct mixture pdf
                p_list = np.ones_like(scores, dtype=float)
                p_list = p_list / p_list.sum()
                criticality = scores if args.criticality_thresh is None else (scores > args.criticality_thresh).astype(float)
                if np.max(criticality) > 3e-1 or np.sum(criticality) > 60:
                    criticality = criticality / (criticality.sum() + 1e-12)
                    pdf_array = (1.0 - args.epsilon) * criticality + args.epsilon * p_list
                else:
                    pdf_array = p_list
                pdf_array = pdf_array / pdf_array.sum()
                # find index of nearest center to the action taken
                action_idx = int(np.argmin(np.linalg.norm(centers - (action[:D].reshape(1, -1)), axis=1)))
                cur_weight = float(p_list[action_idx] / (pdf_array[action_idx]))

            # apply fallback similar to original: if weight too small or too large, use uniform
            if weight * cur_weight < args.min_weight or cur_weight > 1.1:
                pdf_array = p_list
                action_idx = int(np.random.choice(len(pdf_array), p=pdf_array))
                cur_weight = 1.0

            obs_after, reward, terminated, truncated, info = env.step(action)
            steps += 1
            done = bool(terminated) or bool(truncated) or bool(info.get('fallen', False) or info.get('collided', False) or info.get('base_collision', False) or info.get('thigh_collision', False) or info.get('stuck', False))
            crash = bool(info.get('fallen', False) or info.get('collided', False) or info.get('base_collision', False) or info.get('thigh_collision', False) or info.get('stuck', False))
            obs_input = np.asarray(obs, dtype=np.float32).tolist()
            weight *= cur_weight

            if abs(cur_weight - 1) < 1e-5:
                pass
            else: 
                control_step += 1
                weight_step_info[f'{i}'] = cur_weight
                drl_epsilon_step_info[f'{i}'] = action[:D].tolist()
                ndd_step_info[f'{i}'] = float(p_list[action_idx])
                drl_obs_step_info[f'{i}'] = obs_input
    
            if done:
                if crash:
                    episode_data['weight_step_info '] = weight_step_info 
                    episode_data['drl_epsilon_step_info'] = drl_epsilon_step_info 
                    episode_data['ndd_step_info'] = ndd_step_info 
                    episode_data['drl_obs_step_info'] = drl_obs_step_info
                    episode_data['weight_episode'] = float(weight)
                    
                    file_name = f'/mnt/mnt1/linxuan/MyLander_data/d2rl_data/raw_data/crash_{args.batch_idx}_{epoch}.json'
                    with open(file_name,'w') as f:
                        json.dump(convert_to_serializable(episode_data),f)
            i+=1
  
    env.close()
  
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_idx', type=int, default=4)
    parser.add_argument('--model_path', type=str, default='', help='Path to criticality model state_dict')
    parser.add_argument('--criticality_thresh', type=float, default=None)
    parser.add_argument('--epsilon', type=float, default=0.01)
    parser.add_argument('--min_weight', type=float, default=1e-3)
    args = parser.parse_args()
    main(args)