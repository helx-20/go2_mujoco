# terrain_config 字段说明

此文档基于 `deploy_mujoco/terrain_config.yaml`，说明各字段含义与建议值，方便修改与复现实验。

## 顶层字段

- `terrain_action`：与在线地形修改相关的参数。
  - `terrain_types`：支持的地形类型（例如 `bump`）。
  - `min_forward_dist` / `max_forward_dist`（米）：bump 圆心相对于机器人基座的前向距离范围。
  - `max_lateral`（米）：bump 最大横向偏移。
  - `radius_min` / `radius_max`（米）：bump 半径范围。
  - `max_bump_height`（米）：bump 最大高度（绝对高度，通常小于 0.2m）。
  - `no_change_radius`（米）：以机器人为中心，在该半径内禁止修改地形，避免穿透。
  - `terrain_decimation`：用于降低 hfield 修改分辨率的采样因子（整数）。

- `plum_blossom`：用于“梅花桩”风格堆叠地形的参数。
  - `foot_clearance_m`：当增高堆时，若机器人底座低于地形则提升机器人基座的最小间隙（米）。

- `visualization`：渲染/交互控制。
  - `render`：是否在 `TerrainChanger.run()` 中启用渲染（布尔）。
  - `lock_camera`：是否锁定相机。
  - `realtime_sim`：是否以实时速率驱动渲染。

- `logging`：日志开关。
  - `enable_trace`：是否记录详细轨迹（会显著增大日志量）。

- `event_and_reward`：与事件检测和奖励相关的阈值与权重。
  - `failure_flags`：布尔开关控制哪些失败条件应被视作失败事件（如 `fallen`, `collided` 等）。
  - 其它字段如 `fall_height_threshold`, `collision_force_threshold` 等用于事件/奖励判定，单位通常为米或牛顿。

- `termination`：终止条件控制。
  - `terminate_on_fall`, `terminate_on_base_collision`, `terminate_on_stuck`（布尔）。
  - `terrain_edge_margin`（米）：机器人接近地形边界时的安全阈值。

- `observation`：观测相关配置。
  - `include_last_action`, `include_foot_contacts`（布尔）。
  - `local_height_map`：若启用，包含 `enabled`（布尔）、`size_m`（覆盖范围，米）、`resolution_m`（采样分辨率，米）。

## 使用建议

- 在实验中修改 `terrain_action` 时，务必同时更新随机种子与记录参数，以便可复现。
- 如果使用高幅度或大半径的 bump，请增大 `no_change_radius`，或在 `apply_action_vector_with_restore` 中启用恢复机制，避免机器人瞬间穿透地形。
- 对于批量生成地形（评估集中测试），建议使用非交互脚本（`examples/generate_terrain.py`），并将生成的 hfield 保存为二进制文件以便复用。

## Trace 与调试

- 若需要调试地形生成功能，可在 `deploy_mujoco/terrain_params.py` 中启用示例 `__main__` 片段，或使用 `TerrainChanger.run()` 进入可视化循环。

---
如果需要，我可以把此文档转成英文版或把每个字段的默认值与取值范围自动提取到一个表格中。
