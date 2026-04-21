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

from training.utils.test_env import TestEnv, TerrainGymEnv
from training.utils.train_env import TrainEnv
from stable_baselines3.common.monitor import Monitor
import torch
from stable_baselines3 import PPO


model1_path = 'training/models/actor_init.zip'
model2_path = 'training/models/run_value1/best/model_ep80000.zip.policy.pt'
model1 = PPO.load(model1_path, device='cpu')

state_dict = torch.load(model2_path, map_location='cpu')
state_dict = state_dict['policy_state_dict']

print(model1.policy.action_net.bias[-10:])
print(state_dict['action_net.bias'][-10:])