训练目录说明（简体中文）

本目录包含用于在 MuJoCo 中训练 Go2 控制器的实用脚本和说明。主要脚本：

- 离线 PPO 训练: [training/train_controller_ppo_offline.py](training/train_controller_ppo_offline.py)
- 评估/转换/辅助工具: `training/utils/`

快速开始
从仓库根目录运行（示例）：

```bash
python training/train_controller_ppo_offline.py
```

主要特性和行为
- 离线 RL：使用预先收集的数据集训练 PPO 策略，无需在线环境交互。
- 周期性评估：训练过程中定期评估策略表现并保存检查点。
- 日志：支持 TensorBoard 与 CSV 日志。

重要命令/示例
- 离线 PPO 训练：

```bash
python training/train_controller_ppo_offline.py
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

评估与导出
- 若需在部署（`deploy_mujoco`）中使用模型，参考 `training/test_model.py` 或使用 `training/utils/transform_actor_model_to_sb3.py` 将 actor 提取并导出成兼容的 TorchScript 模块。

更多帮助
- 如需我执行一次训练/评估并把平均 reward 报给你，请告诉我希望的参数。
