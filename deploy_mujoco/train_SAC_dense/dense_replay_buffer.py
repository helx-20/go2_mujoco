import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer
from deploy_mujoco.offline_data_utils import filter_chain_for_replay


class FailureReplayBuffer(ReplayBuffer):
    """Replay buffer with failure-biased replay.

    Base behavior:
    - Store selected transitions once.

    Bias behavior:
    - Mark selected rows as dense-selected if episode is failure-relevant
      or cumulative reward is high, then prioritize those rows in sampling.
    """

    def __init__(self, *args, reward_threshold=8.0, dense_sample_ratio=0.7, consecutive_fail_keep_k=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_threshold = float(reward_threshold)
        self.dense_sample_ratio = float(np.clip(dense_sample_ratio, 0.0, 1.0))
        self.consecutive_fail_keep_k = int(max(0, consecutive_fail_keep_k))
        self._pending = [[] for _ in range(self.n_envs)]
        # True means this row was inserted as dense-selected transition.
        self._dense_mask = np.zeros((self.buffer_size,), dtype=np.bool_)

    def _is_failure_info(self, info: dict) -> bool:
        if not isinstance(info, dict):
            return False
        return bool(
            info.get("fallen", False)
            # or info.get("collided", False)
            or info.get("base_collision", False)
            or info.get("thigh_collision", False)
            or info.get("stuck", False)
        )

    def _select_episode_transitions(self, episode_chain):
        return filter_chain_for_replay(
            episode_chain,
            consecutive_fail_keep_k=int(self.consecutive_fail_keep_k),
        )

    def _flush_episode(self, env_idx: int):
        ep = self._pending[env_idx]
        if not ep:
            return

        selected = self._select_episode_transitions(ep)
        if len(selected) == 0:
            self._pending[env_idx] = []
            return

        ep_reward = float(np.sum([float(t["reward"]) for t in selected]))
        has_failure = any(self._is_failure_info(t["info"]) for t in selected)
        has_high_reward = ep_reward >= self.reward_threshold
        dense_selected = bool(has_failure or has_high_reward)

        inserted_indices = []
        for tr in selected:
            info = dict(tr["info"]) if isinstance(tr["info"], dict) else {}
            info["dense_selected"] = dense_selected
            super().add(
                np.expand_dims(tr["obs"], axis=0),
                np.expand_dims(tr["next_obs"], axis=0),
                np.expand_dims(tr["action"], axis=0),
                np.array([tr["reward"]], dtype=np.float32),
                np.array([float(bool(tr["done"]))], dtype=np.float32),
                [info],
            )
            inserted_indices.append((self.pos - 1) % self.buffer_size)

        for idx in inserted_indices:
            self._dense_mask[idx] = dense_selected

        self._pending[env_idx] = []

    def add(self, obs, next_obs, action, reward, done, infos):
        if not isinstance(infos, (list, tuple)):
            infos = [{} for _ in range(self.n_envs)]

        obs = np.asarray(obs)
        next_obs = np.asarray(next_obs)
        action = np.asarray(action)
        reward = np.asarray(reward).reshape(-1)
        done = np.asarray(done).reshape(-1)

        for env_idx in range(self.n_envs):
            info = dict(infos[env_idx]) if env_idx < len(infos) and isinstance(infos[env_idx], dict) else {}
            tr = {
                "obs": obs[env_idx].copy(),
                "next_obs": next_obs[env_idx].copy(),
                "action": action[env_idx].copy(),
                "reward": float(reward[env_idx]),
                "done": float(done[env_idx]),
                "info": info,
            }
            self._pending[env_idx].append(tr)

            if bool(done[env_idx]):
                self._flush_episode(env_idx)

    def sample(self, batch_size, env=None):
        """Prioritize dense-selected rows during sampling.

        - dense_sample_ratio controls fraction sampled from dense_selected=True rows.
        - fallback to uniform sampling if dense pool is empty.
        """
        upper_bound = self.buffer_size if self.full else self.pos
        if upper_bound <= 0:
            return super().sample(batch_size=batch_size, env=env)

        if self.dense_sample_ratio <= 0.0:
            return super().sample(batch_size=batch_size, env=env)

        dense_mask = self._dense_mask[:upper_bound]
        dense_pool = np.flatnonzero(dense_mask)
        if dense_pool.size == 0:
            return super().sample(batch_size=batch_size, env=env)

        n_dense = int(round(batch_size * self.dense_sample_ratio))
        n_dense = max(1, min(batch_size, n_dense))
        # n_dense = min(n_dense, dense_pool.size)

        if dense_pool.size >= n_dense:
            dense_inds = np.random.choice(dense_pool, size=n_dense, replace=False)
        else:
            dense_inds = np.random.choice(dense_pool, size=n_dense, replace=True)

        n_other = batch_size - n_dense
        if n_other > 0:
            other_pool = np.flatnonzero(~dense_mask)
            if other_pool.size >= n_other:
                other_inds = np.random.choice(other_pool, size=n_other, replace=False)
            elif other_pool.size > 0:
                other_inds = np.random.choice(other_pool, size=n_other, replace=True)
            else:
                other_inds = np.random.randint(0, upper_bound, size=n_other)
            batch_inds = np.concatenate([dense_inds, other_inds])
        else:
            batch_inds = dense_inds

        np.random.shuffle(batch_inds)
        return self._get_samples(batch_inds, env=env)
