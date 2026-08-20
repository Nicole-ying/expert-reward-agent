# 设计理由

当前奖励函数已高效完成任务目标，且 score 稳定超过 300。训练反馈显示 `angular_velocity_penalty` 的平均每步贡献仅为 -0.0002，占奖励总量的比例几乎可以忽略，属于冗余组件。从环境事实看，该惩罚用于抑制旋转抖动，但 agent 已学会利用 gate 约束姿态，额外角度速度惩罚已无必要。移除该组件可以简化奖励结构，且不会损害已习得的稳定步态。此为 Level 1 尺度修复——删除无用组件。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Main learning signal: forward velocity, gated by posture health
    forward_vel = next_obs[2]                 # positive = moving forward
    hull_angle = next_obs[0]                  # radians, 0 = vertical
    health = 1.0 / (1.0 + 10.0 * (hull_angle ** 2))   # posture gate
    forward_progress = 1.0 * forward_vel * health

    # Stability: keep torso upright (relaxed safety net)
    tilt_penalty = -2.0 * (hull_angle ** 2)

    # Efficiency: light action cost
    avg_action_power = (action[0]**2 + action[1]**2 + action[2]**2 + action[3]**2) / 4.0
    action_cost = -0.05 * avg_action_power

    total_reward = forward_progress + tilt_penalty + action_cost

    components = {
        'forward_progress': forward_progress,
        'tilt_penalty': tilt_penalty,
        'action_cost': action_cost
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 所有终止条件均有对应前兆信号，进度与效率信号完备，无僵尸组件。
- **behavior**: agent 已学会高速稳定行走，成功到达终点（episode 以成功/失败自然终止，无截断）。
- **signal**: `angular_velocity_penalty` 贡献近乎零，属于冗余组件。
- **level**: Level 1
- **hypothesis**: 移除无用组件可简化奖励结构，避免无关信号干扰后续策略微调，预期性能持平或轻微改善。
- **risk**: 极低，该组件历史贡献微乎其微，移除后不会影响已习得的稳定步态。