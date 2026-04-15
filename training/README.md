Repository-level training helpers for Go2 controller experiments.

This folder contains MuJoCo-based training utilities focused on training
the Go2 controller. The NADE/terrain-agent helpers were removed.

Key files:
- Controller environment: [training/controller_env.py](training/controller_env.py)
- Controller training (PPO): [training/train_controller_ppo.py](training/train_controller_ppo.py)

Quickstart
From repository root (`go2_mujoco`):

```
python training/train_controller_ppo.py --config deploy_mujoco/terrain/configs/go2.yaml
```

Use the provided config to adjust training hyperparameters and the policy
export path. The exported TorchScript policy can be loaded by
`deploy_mujoco`'s `Go2Controller`.

Full workflow to obtain an SB3-initialized actor (`actor_init.zip`)
---------------------------------------------------------------
This project provides utilities to extract a TorchScript actor (if you
have one), convert its weights into a Stable-Baselines3 `PPO` policy
and save an initialized `.zip` for fine-tuning. The following steps
describe the end-to-end process.

1) Inspect a pre-trained artifact (optional)

	 - Use the checker to confirm whether the checkpoint is a TorchScript
		 module or a Python `state_dict`:

		 ```bash
		 python training/utils/check_pretrain.py --path pre_train/go2/go2_controller.pt
		 ```

	 - If the script reports a TorchScript module and prints an output
		 shape (e.g. `(1,12)`), the artifact is deployable as-is.

2) Convert the extracted actor into an SB3 `PPO` policy (`actor_init.zip`)

	 - Use the transformer script which attempts multiple matching
	 	 strategies (numeric-layer mapping). The simplified transformer now
	 	 focuses on extracting numeric MLP layers from a TorchScript/state_dict
	 	 and mapping them into the SB3 policy backbone (`mlp_extractor.policy_net`).
	 	 It writes the SB3 model to `training/model/actor_init.zip` by default:

		 ```bash
	 python training/utils/transform_actor_model_to_sb3.py --actor-pth training/utils/loaded_weights/actor_from_jit.pth --out training/model/actor_init.zip
		 ```

	 - The script will print a concise report of mapped layers. Note:
	 	 the simplified transformer focuses on initializing `mlp_extractor.policy_net`.
	 	 Critic/value networks are not fully recovered by default and should
	 	 be trained/finetuned by PPO.

3) Fine-tune the initialized SB3 model (optional)

	 - Load `training/model/actor_init.zip` as the starting policy for
		 `training/train_controller_ppo.py` (modify the training script to
		 `load` this `.zip` as the base policy), then run PPO fine-tuning
		 in MuJoCo.

Notes on conversion for evaluation (`training/test_model.py`)
-----------------------------------------------------------
- `training/test_model.py` contains a convenience path: if you supply an SB3
	`.zip` as `--controller_path`, the script will attempt to convert the
	policy for use by `deploy_mujoco`'s `Go2Controller`. To keep the exported
	TorchScript minimal and compatible with Go2, the script traces a small
	wrapper that runs `mlp_extractor.policy_net` followed by `action_net` and
	saves the resulting traced module to `training/model/go2_controller_new.pt`.

- This exported module returns only actions (no value/critic outputs) and
	matches the action dimension expected by the Go2 controller.

- If you only want actor weights for SB3 fine-tuning, use the
	`transform_actor_model_to_sb3.py` converter described above to produce
	`training/model/actor_init.zip` and then fine-tune with `train_controller_ppo.py`.

Notes and troubleshooting
- If you have the original Python `state_dict`/checkpoint, prefer
	using it directly instead of extracting from TorchScript — it will
	produce much better initialization coverage.
- The scripts try many heuristics; you should inspect the printed
	matching report. If only actor weights are present in the JIT, the
	critic/value network will be initialized (shared or zero) and must
	be trained/finetuned.
- All helper scripts are now under `training/utils/` and the extracted
	artifacts are stored under `training/utils/loaded_weights/` to keep
	the repo organized.
