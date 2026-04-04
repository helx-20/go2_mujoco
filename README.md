# go2_mujoco

本仓库包含用于在 MuJoCo 中部署与测试 Unitree（Go2）机器人、生成与操控地形的工具与预训练数据。主要用于研究机器人在“角落/边界/突变地形”下的行为与鲁棒性。

主要目录
- `deploy_mujoco/`：用于部署与运行 MuJoCo 场景的脚本（包含 `terrain_params.py`）。
- `pre_train/`：预训练模型与训练检查点（用于快速评估或复现结果）。
- `environment.yml`：建议的 Conda 环境依赖清单（可直接用 `conda env create -f environment.yml` 创建环境）。

快速开始
1. 使用 Conda 创建环境（推荐）：

	 ```bash
	 conda env create -f environment.yml
	 conda activate unitree
	 ```

	 或手动创建并安装依赖：

	 ```bash
	 conda create -n go2_mujoco python=3.8 -y
	 conda activate go2_mujoco
	 pip install mujoco==3.2.3 stable-baselines3==2.4.1 torch==2.3.1+cu121 -r <other-requirements>
	 ```

	 注意：`environment.yml` 中包含 `mujoco==3.2.3`、`stable-baselines3`、`torch` 等库。请根据本机 GPU/OS 调整 `torch` 版本与 CUDA 兼容性，同时确保已正确安装 MuJoCo 及其许可证（并设置相应环境变量，如 `LD_LIBRARY_PATH` 或 `MUJOCO_PY_MUJOCO_PATH`）。

2. 快速运行地形可视化（示例）
- 示例脚本入口：`deploy_mujoco/terrain_params.py` 在模块末尾包含一个 `__main__` 示例，可用来加载场景并以交互方式生成/查看地形。
- 运行该文件前，请确保场景 XML（例如 `deploy_mujoco/robots/go2/scene_terrain.xml`）以及 `terrain_config.yaml` 存在于对应路径。

示例命令：

```bash
python deploy_mujoco/terrain_params.py
```

如果需要以脚本方式生成地形并保存 hfield 或截图，可修改文件末尾的示例代码为非交互流程并添加保存逻辑。

核心模块说明 — `deploy_mujoco/terrain_params.py`
- 类 `TerrainChanger(model, data, action_dims=None, config_file=None)`：用于动态修改 MuJoCo hfield（地形高度场）并保证机器人状态安全。主要能力：
	- 读取 `terrain_config.yaml` 获取地形与 action 配置。
	- 将规范化的动作向量（`action_dims` 可包含 `bump`, `slide_friction`, `solref`）映射为地形变形或物理参数调整。
	- 提供若干地形生成器：`generate_bumps_terrain`, `generate_trig_terrain`, `generate_plum_blossom_piles` 等。
	- 提供安全机制：`enforce_safe_spawn_area`, `_lift_robot_if_needed`, `set_bump_with_restore`（可恢复修改）等，防止地形突变导致机器人穿透或失稳。

主要方法概览：
- `reset(mujoco_data)`：将 hfield 恢复为原始并重置内部状态。
- `run()`：使用 `mujoco.viewer.launch_passive` 进入可视化循环，支持周期性刷新地形（交互/测试用途）。
- `apply_action_vector(action)`：接收扁平 action 向量并应用地形/摩擦/控制参数修改。
- `apply_action_vector_with_restore(action)`：应用修改并返回恢复信息（restore_info），便于事后回滚。
- `generate_bumps_terrain(bumps_array, safe_pos=(0,0), safe_radius=0)`：根据给定 bump 列表初始化地形。
- `generate_trig_terrain(angle_array)`：通过三角函数组合生成连续地形模式。

配置文件 (`terrain_config.yaml`) 要点
- 在 `deploy_mujoco/` 下应包含 `terrain_config.yaml`（示例在仓库中）。关键字段示例包括：
	- `terrain_action`：最小/最大前向距离、横向最大偏移、bump 半径范围、最大 bump 高度等；
	- `plum_blossom`：用于花瓣堆（pile）生成的参数，如 `foot_clearance_m` 等。

如何在训练/评估流程中使用
- 在策略训练或域随机化（domain randomization）流程中，可将 `TerrainChanger` 作为环境的一部分，对观测或物理参数进行在线修改。推荐：在修改地形前使用 `apply_action_vector_with_restore` 获取 `restore_info`，在需要时使用 `set_restore_bump` 还原。

预训练模型
- `pre_train/` 包含若干模型检查点，用于快速测试或作为微调初始权重。具体加载方式请参照训练/评估脚本（仓库内训练脚本路径可能会变化，请搜索 `*.py` 中的模型加载逻辑）。


开发与贡献
- 若要补充示例、修复问题或添加新地形生成器，请提交 PR。建议添加或更新 `deploy_mujoco/terrain_config.yaml` 示例来说明所有可配置字段。

本次更新
- 新增 `docs/terrain_config.md`：字段说明与使用建议。
- 新增 `examples/generate_terrain.py`：非交互脚本，批量生成并保存 hfield（支持生成 PNG 可视化）。
- 新增 `requirements.txt`：从 `environment.yml` 的 pip 部分提取，便于 pip 安装。

下一步建议
- 将 `examples/` 中的脚本扩展为训练与评估一键脚本，并在 `pre_train/` 中添加使用示例。
- 如需，我可以将 README 同步为英文版本或为 `terrain_config.yaml` 生成更详细的表格文档。

许可证
- 当前仓库未在 README 中声明许可，请在确认后补充（例如 MIT/Apache-2.0）。

如果你希望我继续：
- 生成 `terrain_config.yaml` 的字段文档（`docs/terrain_config.md`）。
- 为 `deploy_mujoco/terrain_params.py` 生成 API 风格的使用示例脚本。
- 同步生成英文版 README。

请选择下一步，我将继续完成对应文档或示例代码。