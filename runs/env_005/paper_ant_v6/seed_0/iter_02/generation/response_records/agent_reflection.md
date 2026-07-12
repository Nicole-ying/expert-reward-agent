# Response Record

# 设计理由
我将**姿态约束**从独立惩罚（`posture_penalty`）改为**软健康门控**（soft health gate），乘到前进奖励上，以解决“姿态误差一直存在、惩罚始终活跃但仍大量早期终止”的问题。采用指数衰减因子 `exp(-10 * pose_error)`，在当前姿态误差（约0.07）下门控值约0.5，既能提供显著梯度让 agent 保持直立，又不会完全扼杀学习信号。其他组件（侧向惩罚、高度硬约束）保持不变，本轮只改一个目标组件。

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

    total_reward = forward_reward + lateral_penalty + height_penalty

    components = {
        "forward_reward": forward_reward,
        "lateral_penalty": lateral_penalty,
        "height_penalty": height_penalty
    }
    return float(total_reward), components
```

# 诊断摘要
- **evidence**: terminated=20/20、episode_length≈15.7、posture_penalty 100% 触发性仍然大面积早期终止；height_penalty 仅 6.4%，说明摔倒瞬间高度越界即止，惩罚未能提供预警梯度。
- **behavior**: 四足机器人几乎无法维持直立，在数十步内因高度越界（摔倒）终止，前进速度奖励难以积累。
- **signal**: 持续作用的姿态二次惩罚未能阻止摔倒，缺乏“接近危险姿态时压制主奖励”的早期信号。
- **level**: Level 2 — 数学结构变换（独立惩罚 → 软门控）
- **hypothesis**: 将姿态约束变为乘性门控后，agent 必须保持接近直立状态才能获得正向速度奖励，这会迫使策略在早期就学会稳定的姿态控制，从而减少早期摔倒。
- **risk**: 姿态因子可能在前几个训练阶段将前进奖励压得过低（因子可低至0.1以下），导致学习速度暂时变慢；但指数衰减能保留足够梯度，随着姿态改善门控会自动回升。
