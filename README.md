# go2_mujoco

This repository contains tools and pretrained models for deploying and testing the Unitree Go2 robot in MuJoCo, generating and manipulating terrain. The primary focus is studying robot behavior and robustness in "corner/boundary/abrupt terrain change" scenarios.

## Key Directories
- `training/`: Core training and evaluation pipeline — Filter-BC offline PPO training (`train_controller_ppo_offline.py`), NADE/NDE testing (`test_model.py`), and utility environments (`utils/`).
- `criticality/`: Criticality failure prediction model — binary classifier that predicts whether a terrain configuration will cause locomotion failure.
- `tests/`: Test scripts for NDE (`nde_test_go2.py`), NADE (`nade_test_go2.py`), and rollout recording (`record_rollout.py`).
- `deploy_mujoco/`: MuJoCo simulation infrastructure — terrain changer (`terrain_params.py`), Go2 controller (`terrain/go2_controller.py`), terrain trainer (`terrain_trainer.py`), robot models (`robots/go2/`), and configuration files.
- `pre_train/`: Pretrained models and training checkpoints (for quick evaluation or reproducing results).
- `environment.yml`: Recommended Conda environment dependency list (create the environment with `conda env create -f environment.yml`).

## Quick Start

1. Create the environment with Conda (recommended):

   ```bash
   conda env create -f environment.yml
   conda activate unitree
   ```

   Note: `environment.yml` includes `mujoco==3.2.3`, `stable-baselines3`, `torch`, and other libraries. Adjust the `torch` version and CUDA compatibility according to your GPU/OS.

2. Common commands:

   ```bash
   # Offline PPO training (Filter-BC)
   python training/train_controller_ppo_offline.py

   # NADE evaluation with criticality-guided importance sampling
   python training/test_model.py --nade

   # NDE baseline evaluation
   python tests/nde_test_go2.py

   # Train criticality failure prediction model
   python criticality/stage1/stage1_train.py

   # Interactive terrain visualization
   python deploy_mujoco/terrain_params.py
   ```

## System Architecture

The project has a **hierarchical control structure**:
1. **High-level terrain agent** — modifies the MuJoCo hfield (bumps, friction) via `TerrainChanger` (`deploy_mujoco/terrain_params.py`)
2. **Low-level Go2 locomotion policy** — a pretrained TorchScript model (trained using the Go2 official training script from Isaac Lab) that handles walking, loaded by `Go2Controller`
3. **Training loop** — PPO trains a policy to command the Go2 through dynamically-changing terrain (`training/`)
4. **Criticality prediction** — binary classifier that predicts whether a terrain configuration will cause locomotion failure (`criticality/`)

## Pretrained Models
- `pre_train/` contains model checkpoints for quick testing or as initial weights for fine-tuning. The pretrained Go2 locomotion policy was trained using the Go2 official training script from Isaac Lab.

## Development & Contributions
- To add examples, fix issues, or add new terrain generators, please submit a PR.
