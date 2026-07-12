# 设计理由
当前 agent 已稳定完成地形（score≈330, terminated=20/20, len≈865），主要前进信号为线性 `forward_velocity`，平均约 0.58。略微提高 `forward_reward` 系数至 1.2 可以给前进速度更大权重，在 `health_gate` 保持平衡约束的前提下，鼓励 agent 进一步提高行走速度，可能导致更短的 episode 或更高的累计奖励。这是一个最小尺度干预（Level 1），不改变数学结构，预期不会破坏现有稳定行为。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ============================================================
    # 1. forward_progress: 主学习信号 - 鼓励向前行走（系数微调）
    # ============================================================
    forward_velocity = obs[2]
    forward_reward = 1.2 * forward_velocity

    # ============================================================
    # 2. balance_maintenance: 健康约束 - 保持身体直立
    #    angle_penalty: hinge 惩罚，阈值 0.2 rad
    # ============================================================
    hull_angle = obs[0]
    hull_angular_vel = obs[1]

    angle_threshold = 0.2
    angle_penalty = -1.0 * max(0.0, abs(hull_angle) - angle_threshold)

    angular_vel_penalty = -0.1 * hull_angular_vel ** 2

    balance_penalty = angle_penalty + angular_vel_penalty

    # ============================================================
    # 3. soft_health_gate: 保持原状
    # ============================================================
    gate_k = 3.0
    health_gate = 1.0 / (1.0 + gate_k * abs(hull_angle))
    gated_forward_reward = forward_reward * health_gate

    # ============================================================
    # 4. action_penalty: 能量消耗最小化 - 惩罚大扭矩
    # ============================================================
    action_penalty = -0.03 * (action[0]**2 + action[1]**2 + action[2]**2 + action[3]**2)

    # ============================================================
    # 5. 组合
    # ============================================================
    total_reward = gated_forward_reward + balance_penalty + action_penalty

    components = {
        'forward_reward': forward_reward,
        'health_gate': health_gate,
        'gated_forward_reward': gated_forward_reward,
        'angle_penalty': angle_penalty,
        'angular_vel_penalty': angular_vel_penalty,
        'balance_penalty': balance_penalty,
        'action_penalty': action_penalty
    }

    return float(total_reward), components
```

# 诊断摘要
- **evidence**: 终止率 100%，ep_len≈865，score≈330，forward 平均 0.58，action_penalty 仅占 -1.1%，angle_penalty 未触发，身体极度直立
- **behavior**: 稳定行走并完成地形，步态高效且平衡极稳
- **signal**: 前进速度权重可能略微偏低，存在进一步提升速度而保持稳定的空间
- **level**: Level 1（尺度修复：提高 forward_reward 系数 20%）
- **hypothesis**: 增加前进信号权重会鼓励 agent 在 gate 保护下追求更高速度，从而获得更高累计奖励
- **risk**: 速度过度增加可能触发较大的 hull_angle，被 health_gate 削弱信号，但不会导致摔倒，总体无破坏风险