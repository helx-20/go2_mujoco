"""MuJoCo-based training utilities focused on Go2 controller experiments.

This package contains the MuJoCo Gym environment and training entrypoints
used to train the Go2 controller (controller-only; NADE/terrain helpers
have been removed).
"""

__all__ = ["controller_env", "train_controller_ppo"]
