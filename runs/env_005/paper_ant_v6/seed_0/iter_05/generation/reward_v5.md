# 设计理由
当前 agent 在所有评估回合中迅速摔倒（length≈17.4，terminated 率 100%），表明缺乏维持安全高度的有效信号。已有的 height_gate 在 body_z＜0.25 时直接将 forward_reward 压为 0，导致危险区内彻底丧失学习梯度。本次干预新增一个低高度 hinge 惩罚：`-2.0 * max(0, 0.3 - body_z)`，阈值设在终止边界（0.2）的 60% 处，提供每步约 0.1~0.2 的温和纠正信号（约为主进展信号 forward_reward 的 10%~30%），既警告 agent 远离危险高度，又不会过早切断探索。

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

    # 低高度惩罚（hinge）：身体高度低于 0.3m 时施加温和惩罚
    height_penalty = -2.0 * max(0.0, 0.3 - body_z)

    total_reward = forward_reward + lateral_penalty + action_penalty + height_penalty

    components = {
        "forward_reward": forward_reward,
        "lateral_penalty": lateral_penalty,
        "action_penalty": action_penalty,
        "height_penalty": height_penalty
    }
    return float(total_reward), components
```

# 诊断摘要
- **evidence**: terminated=100%，episode length=17.45，forward_reward active rate 69.3% 但 agent 立刻摔倒。
- **behavior**: agent 无法维持安全高度，迅速终止，几乎没有有效前进。
- **signal**: 缺少低高度纠正信号；现有门控在 0.25 以下切断前向奖励，导致危险区丧失梯度。
- **level**: Level 2 – 新增低高度 hinge 惩罚组件。
- **hypothesis**: 增加温和的高度惩罚能警告 agent 远离 0.2 终止边界，从而延长生存、使前向奖励持续生效并改善表现。
- **risk**: 惩罚系数若过大可能抑制低高度区域的探索，但当前尺度（~0.1‑0.2/步）仅起引导作用，副作用低。