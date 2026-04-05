import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class ReplayBuffer:
    def __init__(self, pos_path, neg_path, pos_ratio=0.5):
        self.pos_buf = np.load(pos_path, allow_pickle=True).tolist()
        self.neg_buf = np.load(neg_path, allow_pickle=True).tolist()
        self.ratio = pos_ratio

    def sample(self, batch_size):
        pos_samples = random.sample(self.pos_buf, int(batch_size * self.ratio))
        pos_inputs = torch.stack([torch.tensor(t['input'].tolist() + t['action'].tolist(), dtype=torch.float) for t in pos_samples])
        pos_next_obs = torch.stack([torch.tensor(t['next_input'], dtype=torch.float) for t in pos_samples])
        pos_rewards = torch.tensor([t['reward'] for t in pos_samples], dtype=torch.float)
        pos_dones = torch.stack([torch.tensor(t['done'], dtype=torch.float) for t in pos_samples])
        neg_samples = random.sample(self.neg_buf, batch_size - int(batch_size * self.ratio))
        neg_inputs = torch.stack([torch.tensor(t['input'].tolist() + t['action'].tolist(), dtype=torch.float) for t in neg_samples])
        neg_next_obs = torch.stack([torch.tensor(t['next_input'], dtype=torch.float) for t in neg_samples])
        neg_rewards = torch.tensor([t['reward'] for t in neg_samples], dtype=torch.float)
        neg_dones = torch.stack([torch.tensor(t['done'], dtype=torch.float) for t in neg_samples])
        inputs = torch.cat([pos_inputs, neg_inputs], dim=0)
        next_obs = torch.cat([pos_next_obs, neg_next_obs], dim=0)
        rewards = torch.cat([pos_rewards, neg_rewards], dim=0)
        dones = torch.cat([pos_dones, neg_dones], dim=0)
        return inputs, next_obs, rewards, dones

    def __len__(self):
        return len(self.pos_buf) + len(self.neg_buf)


class DQN:
    """A minimal DQN wrapper that treats `q_net(state)` as returning a scalar Q-value.

    Expects `q_net` to accept state tensors of shape (B, state_dim) and return (B,1) or (B,).
    The `update(transitions)` method accepts a dict with keys:
      - 'inputs': Tensor (B, state_dim)
      - 'next_inputs': Tensor (B, state_dim)
      - 'rewards': list/array length B
      - 'dones': list/array length B

    This implementation performs a single MSE update toward TD target.
    """

    def __init__(self, q_net, learning_rate=1e-3, gamma=0.95, target_update=50, device='cpu'):
        self.device = torch.device(device)
        self.q_net = q_net.to(self.device)
        self.target_q_net = copy.deepcopy(self.q_net).to(self.device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.gamma = float(gamma)
        self.target_update = int(target_update)
        self.step_count = 0
        self.loss_fn = nn.MSELoss()
        self.action_grid = self.get_action_grid()

    def get_action_grid(self, action_space_size=4):
        action_edges = [np.linspace(-1, 1, num=11) for d in range(action_space_size)]
        D = action_space_size
        grids = np.meshgrid(*[np.arange(10) for _ in range(D)], indexing='ij')
        bins_flat = np.stack([g.reshape(-1) for g in grids], axis=1).astype(np.int64)
        num_actions = bins_flat.shape[0]
        centers = np.zeros((num_actions, D), dtype=np.float32)
        for d in range(D):
            e = action_edges[d]
            b_idx = bins_flat[:, d]
            centers[:, d] = 0.5 * (e[b_idx] + e[b_idx + 1])
        return centers

    def update(self, inputs, next_obs, rewards, dones):
        inputs = inputs.to(self.device)
        next_obs = next_obs.to(self.device)
        rewards = rewards.to(self.device).unsqueeze(1)
        dones = dones.to(self.device).unsqueeze(1)

        self.q_net.train()
        q_vals = self.q_net(inputs)

        q_next_vals = torch.zeros_like(q_vals)
        with torch.no_grad():
            for i in range(next_obs.shape[0]):
                cur_obs = next_obs[i].repeat(self.action_grid.shape[0], 1)
                cur_act = torch.from_numpy(self.action_grid).to(self.device)
                cur_next_input = torch.cat([cur_obs, cur_act], dim=1)
                cur_next_q = self.target_q_net(cur_next_input)
                q_next_vals[i] = torch.max(cur_next_q, dim=0)[0]
        q_targets = dones + self.gamma * q_next_vals * (1 - dones)

        loss = self.loss_fn(q_vals, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_count += 1
        if self.target_update > 0 and (self.step_count % self.target_update == 0):
            try:
                self.target_q_net.load_state_dict(self.q_net.state_dict())
            except Exception:
                pass

        return float(loss.item())
