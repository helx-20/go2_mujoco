# NDE / NADE tests for go2_mujoco

This folder contains example scripts to run NDE (naive deterministic evaluation) and NADE-like (weighted evaluation) tests for the Go2 MuJoCo environment. They are adapted from the MyLander project's `nde_test.py` and `nade_test.py`.

Files
 - `nde_test_go2.py` — run large-scale rollouts where terrain actions are sampled uniformly from the environment's `action_space` (NDE: uniform exploration over controllable terrain parameters).
 - `nade_test_go2.py` — run rollouts where terrain actions are biased using a provided criticality model: at each decision the script samples multiple candidate terrain actions, scores them with the criticality model, and selects by probability (or greedily) according to scores. Use `--model_path` to pass a PyTorch criticality model.
  
Output
- Both scripts support `--out` to save results as a NumPy `.npy` file and periodically save during runs. Defaults:
	- NDE: `results/nde_crashes.npy` (array of 0/1 crash flags)
	- NADE: `results/nade_weighted.npy` (array of weighted failure values)


How to run
1. Create the conda environment (see repository root README) and install dependencies (or use `requirements.txt`).
2. From repository root run examples, e.g.:

```bash
python -m tests.nde_nade.nde_test_go2 --episodes 1000
python -m tests.nde_nade.nade_test_go2 --episodes 1000 --model_path path/to/crit_model.pt
```

Notes
- Both scripts use the `TerrainTrainer` and `TerrainGymEnv` provided in `deploy_mujoco/` and assume the default go2 config (`terrain/configs/go2.yaml`) and `deploy_mujoco/terrain_config.yaml` are present.
- These are intended as starting templates; adapt logging, GPU device, and save locations to your needs.
