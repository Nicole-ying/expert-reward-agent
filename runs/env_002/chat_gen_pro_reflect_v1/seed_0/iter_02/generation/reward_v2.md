# 设计理由
angle_penalty 的 active_rate 为 0%，说明当前 0.5 rad 阈值从未触发，无法对倾斜行为提供任何直接负反馈，导致部分 episode 可能仍会发生摔倒。我将该组件从“几乎不生效”变为“更早警告”：把 hinge 阈值降至 0.2 rad，并把系数从 -2.0 调整为 -1.0，使 per‑step 惩罚预期约在 0–0.1 量级（主前进信号约 0.5），既能提供梯度又不至于过度抑制前进探索。这样在身体开始倾斜时就给予温和惩罚，配合已有的 health_gate 衰减，应能提升稳定性。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ============================================================
    # 1. forward_progress: 主学习信号 - 鼓励向前行走
    # ============================================================
    forward_velocity = obs[2]
    forward_reward = 1.0 * forward_velocity

    # ============================================================
    # 2. balance_maintenance: 健康约束 - 保持身体直立
    #    angle_penalty: hinge 惩罚，阈值降至 0.2 rad (~11.5°)
    #    系数 -1.0 使 per‑step 惩罚与前进奖励尺度匹配
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
    # 4. 组合
    # ============================================================
    total_reward = gated_forward_reward + balance_penalty

    components = {
        'forward_reward': forward_reward,
        'health_gate': health_gate,
        'gated_forward_reward': gated_forward_reward,
        'angle_penalty': angle_penalty,
        'angular_vel_penalty': angular_vel_penalty,
        'balance_penalty': balance_penalty
    }

    return float(total_reward), components
```

# 诊断摘要
- **evidence**: angle_penalty active_rate=0%，但 low‑score episodes 存在，说明倾斜可能未被及时察觉；主信号 gated_forward 约 0.5/步。
- **behavior**: agent 能走较远但有时摔倒，导致平均分未达 300。
- **signal**: 角度约束缺失，需要在倾斜初期提供直接负反馈。
- **level**: Level 2（阈值从 0.5→0.2，激活原本休眠的惩罚，数学结构不变但功能质变）。
- **hypothesis**: 降低 hinge 阈值并匹配系数，将给予更早的纠正信号，减少摔倒 episode，提升平均得分。
- **risk**: 惩罚可能过度限制探索，导致前进速度下降或畏缩不前。