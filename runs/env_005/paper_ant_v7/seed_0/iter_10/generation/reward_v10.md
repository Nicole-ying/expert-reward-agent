# 设计理由

本轮修改 `lateral_velocity_penalty` 组件。第9轮加入该惩罚后，前进速度表现急剧下降（score 从 1414 暴跌至 370），实际横向速度惩罚 active_rate=100%，表明每步都被施加惩罚，严重打压了四足步态中必要的小幅侧向摆动，导致策略牺牲前进速度来回避惩罚。将全局连续惩罚改为门控 hinge：仅在横向速度超出阈值（0.3）时施加轻量惩罚，允许正常步态中的小幅度侧向偏移不受干扰，预期释放前进速度，使 score 恢复到第8轮水平甚至更高。系数从 -0.2 降至 -0.1，并在 threshold 以下完全不惩罚，避免阻碍探索。干预层级 Level 2，结构变换：全时惩罚 → 局部门控。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 信号提取 ----------
    body_z = obs[0]
    quat_x = obs[2]
    quat_y = obs[3]
    body_x_velocity = obs[13]
    body_y_velocity = obs[14]
    torque_penalty = sum([a**2 for a in action])

    # ---------- 身体直立投影 ----------
    body_up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)

    # ---------- 高度安全门控 ----------
    low_factor = max(0.0, (body_z - 0.2) / 0.1)
    high_factor = max(0.0, (1.0 - body_z) / 0.1)
    height_factor = min(low_factor, high_factor)
    height_reward = 0.1 * height_factor

    # ---------- 高度边界前兆惩罚（保留，虽然已接近失效但仍作为安全网） ----------
    low_penalty = -0.3 * max(0.0, (0.25 - body_z) / 0.05)
    high_penalty = -0.3 * max(0.0, (body_z - 0.95) / 0.05)
    height_boundary_penalty = low_penalty + high_penalty

    # ---------- 主学习信号：凸化前进速度 ----------
    forward_reward = 1.0 * body_x_velocity + 0.5 * max(0.0, body_x_velocity)**2

    # ---------- 横向速度惩罚（门控hinge，避免阻碍正常步态中的小幅侧向摆动） ----------
    lateral_speed = abs(body_y_velocity)
    lateral_threshold = 0.3
    lateral_velocity_penalty = -0.1 * max(0.0, lateral_speed - lateral_threshold)

    # ---------- 力矩效率约束 ----------
    action_cost = -0.05 * torque_penalty

    # ---------- 总奖励 ----------
    total_reward = (forward_reward + height_reward + height_boundary_penalty
                    + lateral_velocity_penalty + action_cost)

    components = {
        "forward_reward": forward_reward,
        "height_reward": height_reward,
        "height_boundary_penalty": height_boundary_penalty,
        "lateral_velocity_penalty": lateral_velocity_penalty,
        "action_cost": action_cost
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 缺少直立姿态直接奖励，高度边界前兆几乎僵尸（active_rate≈0%）；当前优先修复 score 暴跌根因。
- **behavior**: agent 为躲避每步侧向惩罚而牺牲前进速度，导致前进组件输出剧烈下降。
- **signal**: 横向速度惩罚是全局连续惩罚，抑制了四足典型步态中必需的小幅侧向摆动。
- **level**: Level 2（全局惩罚 → 局部门控 hinge）
- **hypothesis**: 门控形式允许安全侧向偏摆，释放前进速度，使 score 向 iter8 的 1414 水平恢复。
- **risk**: 侧向速度在某些时刻可能超过 threshold 但仍得到适度惩罚，不会导致严重侧移；若阈值设置不当，可后续微调。