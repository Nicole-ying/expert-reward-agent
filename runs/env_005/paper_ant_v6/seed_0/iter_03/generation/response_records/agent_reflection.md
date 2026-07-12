# Response Record

# 设计理由
agent 学会了快速前进（forward_reward 每步约1.27）但外部得分仍然很低，推测缺少对控制代价（动作幅度过大）的反馈。本轮新增一个 `action_penalty` 组件，用动作的平方和惩罚过大的扭矩，系数 -0.1，预期每步惩罚约 -0.07~-0.2，占主前进信号的 5–15%，符合尺度校准。其他组件保持不变，以检验该惩罚能否降低能耗并提升外部得分。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取关键信号
    body_z = next_obs[0]
    quat_x = next_obs[2]
    quat_y = next_obs[3]
    quat_z = next_obs[4]
    vx = next_obs[13]
    vy = next_obs[14]

    # 姿态健康因子：用指数衰减将四元数虚部平方和映射到 (0,1]
    pose_error = quat_x**2 + quat_y**2 + quat_z**2
    posture_factor = 2.718281828 ** (-10.0 * pose_error)

    # 前进奖励（基础量 × 姿态门控）
    base_forward = max(0.0, vx)
    forward_reward = 1.0 * base_forward * posture_factor

    # 侧向速度惩罚
    lateral_penalty = -0.5 * (vy ** 2)

    # 高度硬约束（仅在越出安全范围时激活，作为后备保护）
    height_exceed = max(0.0, 0.2 - body_z) + max(0.0, body_z - 1.0)
    height_penalty = -10.0 * height_exceed

    # 新增：动作幅度惩罚（控制代价）
    action_penalty = -0.1 * sum(a ** 2 for a in action)

    total_reward = forward_reward + lateral_penalty + height_penalty + action_penalty

    components = {
        "forward_reward": forward_reward,
        "lateral_penalty": lateral_penalty,
        "height_penalty": height_penalty,
        "action_penalty": action_penalty
    }
    return float(total_reward), components
```

# 诊断摘要
- **evidence**: forward_reward 均值 1191（活跃率 96%）但 score 仅 -14.4，策略快速前进却未提升外部得分。
- **behavior**: 15% 摔倒，85% 存活至 938 步上限；主要维持高速前进，侧向速度受控，高度安全。
- **signal**: 缺失控制代价（能耗）的反馈，使策略可能以不必要的大扭矩换取速度，损害外部得分（含隐含控制代价）。
- **level**: Level 2 — 新增 action_penalty 组件（动作平方和惩罚），以对齐隐含的控制成本信号。
- **hypothesis**: 加入动作幅度惩罚后，策略将减少过度扭矩，实现更节能、更稳定的运动，从而提升外部得分。
- **risk**: 惩罚过强可能抑制前进探索，导致速度骤降；若发生，则下一轮可降低系数或改用关节速度惩罚。
