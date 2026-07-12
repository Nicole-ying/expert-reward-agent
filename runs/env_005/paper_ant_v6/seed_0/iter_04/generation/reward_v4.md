# 设计理由
将原先几乎不激活的高度硬惩罚（active_rate 4.8%）替换为一个连续的高度健康门控因子，乘入前进奖励，让身体高度从安全区（≥0.55m）过渡到危险区（0.25m）时平滑衰减奖励。这样 agent 在行走中一旦高度下降就能立即感知到信号减弱，从而学会主动维持安全高度，避免突发性摔倒，而不再依赖几乎无梯度的一次性惩罚。线性 ramp 从 0.25 到 0.55 保证了在 0.35m 时 gate 仍有约 0.33，不会使学习信号完全塌缩。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    body_z = next_obs[0]
    quat_x = next_obs[2]
    quat_y = next_obs[3]
    quat_z = next_obs[4]
    vx = next_obs[13]
    vy = next_obs[14]

    # 姿态健康因子
    pose_error = quat_x**2 + quat_y**2 + quat_z**2
    posture_factor = 2.718281828 ** (-10.0 * pose_error)

    # 高度健康门控：从危险区 (0.25m) 到安全区 (0.55m) 线性提升
    height_gate = 0.0
    if body_z >= 0.55:
        height_gate = 1.0
    elif body_z <= 0.25:
        height_gate = 0.0
    else:
        height_gate = (body_z - 0.25) / 0.3

    # 前进奖励（基础量 × 姿态门控 × 高度门控）
    base_forward = max(0.0, vx)
    forward_reward = 1.0 * base_forward * posture_factor * height_gate

    # 侧向速度惩罚
    lateral_penalty = -0.5 * (vy ** 2)

    # 动作幅度惩罚
    action_penalty = -0.1 * sum(a ** 2 for a in action)

    total_reward = forward_reward + lateral_penalty + action_penalty

    components = {
        "forward_reward": forward_reward,
        "lateral_penalty": lateral_penalty,
        "action_penalty": action_penalty
    }
    return float(total_reward), components
```

# 诊断摘要
- **evidence**: 全部 episode 均因高度出界终止，平均存活仅 20.85 步，高度硬惩罚 active_rate 仅 4.8%
- **behavior**: agent 在极短时间内快速前进后摔倒，姿态恶化导致 height_penalty 几乎来不及触发
- **signal**: 缺少安全高度附近的连续预警信号，agent 无法学习在高度下降时主动修正
- **level**: Level 2（将硬约束惩罚替换为 soft health gate）
- **hypothesis**: 高度门控提供平滑梯度，使 agent 学会在高度下降初期提升身体，从而延长存活时间并维持稳定前进
- **risk**: 若高度门控阈值过严可能抑制探索，但 ramp 设计保留了 0.25-0.55 的过渡区间，风险较低