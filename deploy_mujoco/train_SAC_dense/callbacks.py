import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

import csv
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback


class DenseTrainingLogger(BaseCallback):
    def __init__(self, out_dir: str, save_every_steps: int = 2000, smooth_window: int = 20, checkpoint_every_steps: int = 10000, checkpoint_start_after_steps: int = 0, checkpoint_dir: str = None, verbose: int = 0):
        super().__init__(verbose)
        self.out_dir = out_dir
        self.save_every_steps = int(max(1, save_every_steps))
        self.smooth_window = int(max(1, smooth_window))
        self.checkpoint_every_steps = int(max(1, checkpoint_every_steps))
        self.checkpoint_start_after_steps = int(max(0, checkpoint_start_after_steps))
        self.checkpoint_dir = checkpoint_dir or out_dir

        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_timesteps = []
        self.actor_losses = []
        self.critic_losses = []
        self.loss_timesteps = []

        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if isinstance(info, dict) and "episode" in info:
                self.episode_rewards.append(float(info["episode"]["r"]))
                self.episode_lengths.append(float(info["episode"].get("l", 0)))
                self.episode_timesteps.append(int(self.num_timesteps))

        if hasattr(self.model, "logger"):
            log_dict = self.model.logger.name_to_value
            wrote_loss = False
            if "train/actor_loss" in log_dict:
                self.actor_losses.append(float(log_dict["train/actor_loss"]))
                wrote_loss = True
            if "train/critic_loss" in log_dict:
                self.critic_losses.append(float(log_dict["train/critic_loss"]))
                wrote_loss = True
            if wrote_loss:
                self.loss_timesteps.append(int(self.num_timesteps))

        if self.num_timesteps % self.save_every_steps == 0:
            self._dump_metrics_csv("metrics_partial.csv")
            self._save_combined_plot("training_curves_partial.png")

        if self.num_timesteps % self.checkpoint_every_steps == 0 and self.num_timesteps >= self.checkpoint_start_after_steps:
            ckpt_path = os.path.join(self.checkpoint_dir, f"checkpoint_step_{self.num_timesteps}.zip")
            self.model.save(ckpt_path)

        return True

    def _on_training_end(self) -> None:
        self._save_plot(self.episode_rewards, "Episode Reward", "episode_reward.png")
        if len(self.actor_losses) > 0:
            self._save_plot(self.actor_losses, "Actor Loss", "actor_loss.png")
        if len(self.critic_losses) > 0:
            self._save_plot(self.critic_losses, "Critic Loss", "critic_loss.png")

        self._dump_metrics_csv("metrics.csv")
        self._save_combined_plot("training_curves.png")

    def _smooth(self, values):
        arr = np.asarray(values, dtype=np.float32)
        if arr.size < self.smooth_window:
            return arr
        kernel = np.ones((self.smooth_window,), dtype=np.float32) / float(self.smooth_window)
        return np.convolve(arr, kernel, mode="valid")

    def _save_plot(self, values, title: str, filename: str):
        if len(values) == 0:
            return
        fig = plt.figure()
        plt.plot(values, linewidth=1.0)
        smoothed = self._smooth(values)
        if smoothed.size > 0 and smoothed.size != len(values):
            offset = len(values) - smoothed.size
            plt.plot(np.arange(offset, len(values)), smoothed, linewidth=2.0)
        plt.title(title)
        plt.grid(True)
        fig.savefig(os.path.join(self.out_dir, filename), dpi=140)
        plt.close(fig)

    def _dump_metrics_csv(self, filename: str):
        csv_path = os.path.join(self.out_dir, filename)
        max_len = max(len(self.episode_rewards), len(self.actor_losses), len(self.critic_losses))
        if max_len == 0:
            return

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "episode_timestep",
                "episode_reward",
                "episode_length",
                "loss_timestep",
                "actor_loss",
                "critic_loss",
            ])
            for i in range(max_len):
                writer.writerow([
                    self.episode_timesteps[i] if i < len(self.episode_timesteps) else "",
                    self.episode_rewards[i] if i < len(self.episode_rewards) else "",
                    self.episode_lengths[i] if i < len(self.episode_lengths) else "",
                    self.loss_timesteps[i] if i < len(self.loss_timesteps) else "",
                    self.actor_losses[i] if i < len(self.actor_losses) else "",
                    self.critic_losses[i] if i < len(self.critic_losses) else "",
                ])

    def _save_combined_plot(self, filename: str):
        if len(self.episode_rewards) == 0 and len(self.actor_losses) == 0 and len(self.critic_losses) == 0:
            return

        fig, axes = plt.subplots(3, 1, figsize=(8, 10))

        if len(self.episode_rewards) > 0:
            axes[0].plot(self.episode_timesteps, self.episode_rewards, linewidth=1.0, label="reward")
            smoothed = self._smooth(self.episode_rewards)
            if smoothed.size > 0 and smoothed.size != len(self.episode_rewards):
                offset = len(self.episode_rewards) - smoothed.size
                axes[0].plot(self.episode_timesteps[offset:], smoothed, linewidth=2.0, label=f"reward_ma{self.smooth_window}")
            axes[0].set_title("Episode Reward")
            axes[0].grid(True)
            axes[0].legend(loc="best")

        if len(self.actor_losses) > 0:
            x_actor = self.loss_timesteps[:len(self.actor_losses)] if len(self.loss_timesteps) >= len(self.actor_losses) else np.arange(len(self.actor_losses))
            axes[1].plot(x_actor, self.actor_losses, linewidth=1.0, label="actor_loss")
            axes[1].set_title("Actor Loss")
            axes[1].grid(True)
            axes[1].legend(loc="best")

        if len(self.critic_losses) > 0:
            x_critic = self.loss_timesteps[:len(self.critic_losses)] if len(self.loss_timesteps) >= len(self.critic_losses) else np.arange(len(self.critic_losses))
            axes[2].plot(x_critic, self.critic_losses, linewidth=1.0, label="critic_loss")
            axes[2].set_title("Critic Loss")
            axes[2].grid(True)
            axes[2].legend(loc="best")

        for ax in axes:
            ax.set_xlabel("Timesteps")

        fig.tight_layout()
        fig.savefig(os.path.join(self.out_dir, filename), dpi=150)
        plt.close(fig)
