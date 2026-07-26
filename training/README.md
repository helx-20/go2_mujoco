# Training Directory

This directory contains utility scripts and instructions for training the Go2 controller in MuJoCo. Key scripts:

- Offline PPO training: [training/train_controller_ppo_offline.py](training/train_controller_ppo_offline.py)
- Evaluation/conversion/helper tools: `training/utils/`

## Quick Start
Run from the repository root (example):

```bash
python training/train_controller_ppo_offline.py
```

## Key Features & Behavior
- **Offline RL**: Train a PPO policy using pre-collected datasets, no online environment interaction required.
- **Periodic Evaluation**: Policy performance is evaluated periodically during training and checkpoints are saved.
- **Logging**: Supports TensorBoard and CSV logging.

## Important Commands / Examples
- Offline PPO training:

```bash
python training/train_controller_ppo_offline.py
```

- Run TensorBoard to view training/evaluation curves:

```bash
tensorboard --logdir training/logs/<run_name> --port 6006
```

## TensorBoard FAQ (protobuf compatibility)
- If you encounter an error like the following when opening TensorBoard in a browser:

  TypeError: MessageToJson() got an unexpected keyword argument 'including_default_value_fields'

  This is a common issue with `protobuf` version incompatibility with the TensorBoard hparams plugin. The fix: install a compatible `protobuf` version (e.g. 3.20.3) in your conda/env environment:

```bash
conda activate unitree            # or the environment you actually use
python -m pip install --upgrade --force-reinstall "protobuf==3.20.3"
pkill -f tensorboard || true
tensorboard --logdir training/logs/<run_name> --port 6006
```

## Evaluation & Export
- To use a model in deployment (`deploy_mujoco`), refer to `training/test_model.py` or use `training/utils/transform_actor_model_to_sb3.py` to extract the actor and export it as a compatible TorchScript module.

## More Help
- For a training/evaluation run with average reward reporting, please specify the desired parameters.
