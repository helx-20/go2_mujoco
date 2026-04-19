训练目录说明（简体中文）

本目录包含用于在 MuJoCo 中训练 Go2 控制器的实用脚本和说明。主要脚本：

- 控制器环境: [training/controller_env.py](training/controller_env.py)
- 训练主脚本: [training/train_controller_ppo.py](training/train_controller_ppo.py)
- 评估/转换/辅助工具: `training/utils/`

快速开始
从仓库根目录运行（示例）：

```bash
python training/train_controller_ppo.py --run_name debug_run --total_timesteps 50000 --num_envs 16 --n_steps 128 --batch_size 2048
```

主要特性和行为
- 预训练加载：如果你传入 `--pretrain` 指向一个 SB3 `.zip`，脚本不会直接用 `SB3PPO.load(..., env=...)` 载入带有旧训练状态的模型。脚本会在 CPU 上加载该 `.zip` 仅用于提取 `policy.state_dict()`，然后创建一个干净的 `PPO(...)` 并按 name/shape 将权重拷贝进去（避免携带旧的 rollout buffer 或 optimizer 状态）。
- 周期性评估：脚本内置周期性评估回调（基于 `PolicyNetEvalCallback`），会在每次评估时保存 `last_model.zip`，并在发现更好表现时保存 `best_model.zip`，保存在 `--out` 指定目录下的 `best/` 子目录。
- 日志：支持 TensorBoard 与 CSV 日志（通过 `--tensorboard_log` 指定目录），训练日志默认写入 `training/logs/<run_name>`。
- Rollout buffer：脚本会确保 `RolloutBuffer.buffer_size == n_steps`（SB3 的设计），并创建/替换不匹配的 buffer，避免 buffer 未被填满导致的训练断言错误。

重要命令/示例
- 使用预训练权重进行微调（脚本会把 `.zip` 里的 policy 拷贝到新模型）：

```bash
python training/train_controller_ppo.py --pretrain training/model/actor_init.zip --run_name finetune_run --total_timesteps 200000
```

- 跳过预训练（强制创建全新模型）：

```bash
python training/train_controller_ppo.py --pretrain /dev/null --run_name fresh_run
```

- 运行 TensorBoard 查看训练/评估曲线：

```bash
tensorboard --logdir training/logs/<run_name> --port 6006
```

TensorBoard 常见问题（protobuf 兼容性）
- 如果在浏览器打开 TensorBoard 时遇到类似：

  TypeError: MessageToJson() got an unexpected keyword argument 'including_default_value_fields'

  这是 `protobuf` 版本与 TensorBoard 插件（hparams）不兼容的常见问题。解决办法：在你的 conda/env 环境中安装兼容的 `protobuf`（例如 3.20.3）：

```bash
conda activate unitree            # 或你实际使用的环境
python -m pip install --upgrade --force-reinstall "protobuf==3.20.3"
pkill -f tensorboard || true
tensorboard --logdir training/logs/<run_name> --port 6006
```

调试与超参数建议
- 推荐在多进程环境下使用较小的 `n_steps`（例如 128）并搭配较多的 `num_envs`（例如 8-16），以保持稳定和较小的内存占用。确保 `batch_size <= n_steps * num_envs`，并尽量使 `total_rollout = n_steps * num_envs` 能被 `batch_size` 整除以获得均匀 minibatch 划分。
- 如果训练过程中出现 buffer/索引断言错误，通常是因为加载了带有旧内部状态的 `.zip`；本仓库已避免直接带入该状态，但如果你自行修改了加载逻辑，请使用脚本内的“创建干净模型 + 拷贝权重”流程。

评估与导出
- `PolicyNetEvalCallback` 在每次评估时会调用内部的 `evaluate_policy()`（使用 `TestEnv`/`TerrainGymEnv`），打印每回合奖励并保存 `last_model.zip` 与 `best_model.zip`（若更好）。
- 若需在部署（`deploy_mujoco`）中使用模型，参考 `training/test_model.py` 或使用 `training/utils/transform_actor_model_to_sb3.py` 将 actor 提取并导出成兼容的 TorchScript 模块。

更多帮助
- 查看 `training/train_controller_ppo.py` 中的命令行参数说明（`--help`）。
- 如需我执行一次短训练/评估并把平均 reward 报给你，请告诉我希望的参数（`--num_envs`、`--n_steps`、`--total_timesteps`、`--run_name`）。

---
更新日志：简化了预训练加载流程，增加了周期性评估保存最新模型，修复并保护了 rollout buffer 的创建逻辑。
