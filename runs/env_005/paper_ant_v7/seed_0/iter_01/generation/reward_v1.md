# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 信号提取 ----------
    body_z = obs[0]
    quat_x = obs[2]
    quat_y = obs[3]
    body_x_velocity = obs[13]
    # 注意：action 是 8 维力矩，范围 [-1, 1]
    torque_penalty = sum([a**2 for a in action])  # 默认可迭代，但保险起见写成列表

    # ---------- 主学习信号：向前速度 ----------
    # 线性正奖励，鼓励更大的向前速度
    forward_reward = 1.0 * body_x_velocity

    # ---------- 身体高度生存约束（hinge 惩罚） ----------
    # 健康区间为 [0.3, 0.9]，接近边界 0.2 或 1.0 时给予线性惩罚
    low_danger = max(0.0, 0.3 - body_z)
    high_danger = max(0.0, body_z - 0.9)
    height_penalty = -2.0 * (low_danger + high_danger)

    # ---------- 直立姿态约束（二次惩罚） ----------
    # 身体直立投影 body_up_z = 1 - 2*(quat_x^2 + quat_y^2)，越接近 1 越好
    body_up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)
    upright_error = 1.0 - body_up_z  # 0 表示完全直立
    upright_penalty = -2.0 * (upright_error**2)

    # ---------- 力矩效率约束（极小权重，防极端动作） ----------
    action_cost = -0.01 * torque_penalty

    # ---------- 总奖励 ----------
    total_reward = forward_reward + height_penalty + upright_penalty + action_cost

    components = {
        "forward_reward": forward_reward,
        "height_penalty": height_penalty,
        "upright_penalty": upright_penalty,
        "action_cost": action_cost
    }
    return float(total_reward), components
```

# reward_v1 设计说明

- **selected task_family / dynamics_subtype**：`locomotion_continuous_control` / `multi_legged_body_locomotion`（四足机器人持续向前行走）。
- **selected reward roles**：
  - `forward_velocity_reward`（主学习信号，必须）
  - `body_height_survival`（安全约束，必须）
  - `upright_orientation_penalty`（姿态约束，必须）
  - `joint_torque_regularization`（轻量效率/防抖约束，因动作维度≥6 且 NaN 风险而纳入 v1）
- **role_to_signal_mapping**：
  - `forward_velocity_reward`：`obs[13]` (body_x_velocity) → 线性正奖励
  - `body_height_survival`：`obs[0]` (body_z) → hinge 线性惩罚，危险边界为 0.3 与 0.9
  - `upright_orientation_penalty`：`obs[2], obs[3]` (quat_x, quat_y) → 计算 `body_up_z`，二次惩罚 1 - body_up_z
  - `joint_torque_regularization`：`action[0:8]` → 极小平方和惩罚
- **每个 role 选择的 formula operator**：
  - 主进度：`dense_state_signal` 线性正形式 `w * signal`
  - 高度约束：`dense_state_signal` 的 hinge 形式（仅在越界时生效，边界内为 0）
  - 姿态约束：`quadratic_penalty`（防止大幅倾斜，持续给予梯度）
  - 动作约束：`quadratic_penalty`（极小权重，避免极端力矩）
- **excluded roles 及原因**：
  - `terminal_success_reward`、`terminal_failure_penalty`：无显式成功/失败标志
  - `foot_contact_coordination`：无接触信号
  - `distance_from_origin`：位置信号被禁止
  - 横向/垂直/角速度惩罚、复杂门控：留到后续迭代，v1 先建立基本向前策略。
- **为什么没有使用 terminal_success_reward / terminal_failure_penalty**：环境未提供 `info` 中的成功或失败标志，且不允许从终止状态间接推断奖励，只能通过观测信号设计稠密约束。
- **哪些职责留到后续迭代**：横向漂移、垂直跳动、角速度平滑、更高级的步态协调。若训练后期出现过度抖动或能耗过高，可逐步上调 `action_cost` 权重。
- **训练后应观察的 failure modes**：
  - 警告：如果策略收敛到静止站立（`body_x_velocity`≈0），说明高度或姿态惩罚压制了前进奖励，需降低相关权重。
  - 频繁终止（高度＜0.2）：检查 hinge 边界是否足够警告（当前从 0.3 起罚），或增加高度惩罚权重。
  - 侧翻或倒立前进（`body_up_z` 持续低于 0.8）：加大 `upright_penalty` 系数。
  - 关节抖动或 NaN：`action_cost` 可防止部分极端动作，若仍出现 NaN，需全局梯度裁剪或更低的力矩权重。