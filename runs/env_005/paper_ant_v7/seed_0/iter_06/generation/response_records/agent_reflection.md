# Response Record

# 设计理由
当前 reward 成功引导了高速前进和稳定姿态，但终止率仍有 75%（15/20 为 failure termination），主要怀疑是身体高度偶尔越界导致瞬时终止，而此前没有任何梯度信号。现有 `height_reward` 只在安全区内给正向奖励，边界处奖励归零但不施加惩罚，形成了 **奖励悬崖**：agent 在高度接近 0.2 或 1.0 时得不到负面反馈，一旦越过立刻终止。因此需要增加一个 **高度边界前兆软惩罚**，用 hinge 形式对接近危险区的状态施加轻量负奖励，在不破坏现有成功策略的前提下提供“远离边界”的梯度。

本修改保持其他组件不变，仅新增 `height_boundary_penalty`，属于 **Level 2 结构变换**（添加缺失的预警信号）。系数经校准：主信号 per-step ≈ 4.7，新 penalty 在激活区域最多 -0.3，且只在极端高度出现，平均负担远低于 0.5× 主信号，不会压垮学习。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 信号提取 ----------
    body_z = obs[0]
    quat_x = obs[2]
    quat_y = obs[3]
    body_x_velocity = obs[13]
    torque_penalty = sum([a**2 for a in action])

    # ---------- 身体直立投影 ----------
    body_up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)

    # ---------- 高度安全门控（独立正向生存奖励） ----------
    low_factor = max(0.0, (body_z - 0.2) / 0.1)
    high_factor = max(0.0, (1.0 - body_z) / 0.1)
    height_factor = min(low_factor, high_factor)
    height_reward = 0.1 * height_factor

    # ---------- 高度边界前兆惩罚（防止突然终止） ----------
    low_penalty = -0.3 * max(0.0, (0.25 - body_z) / 0.05)
    high_penalty = -0.3 * max(0.0, (body_z - 0.95) / 0.05)
    height_boundary_penalty = low_penalty + high_penalty

    # ---------- 主学习信号：凸化前进速度 ----------
    forward_reward = 1.0 * body_x_velocity + 0.5 * max(0.0, body_x_velocity)**2

    # ---------- 直立姿态约束（hinge 惩罚，系数降低） ----------
    upright_error = max(0.0, 0.7 - body_up_z)
    upright_penalty = -0.3 * upright_error

    # ---------- 力矩效率约束 ----------
    action_cost = -0.05 * torque_penalty

    # ---------- 总奖励 ----------
    total_reward = (forward_reward + height_reward + height_boundary_penalty
                    + upright_penalty + action_cost)

    components = {
        "forward_reward": forward_reward,
        "height_reward": height_reward,
        "height_boundary_penalty": height_boundary_penalty,
        "upright_penalty": upright_penalty,
        "action_cost": action_cost
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 终止主要来自身体高度越界，但代码中缺少对高度边界的前兆软信号，只有安全区奖励，造成奖励悬崖。
- **behavior**: agent 已经学会高速前进且多数时间高度健康，但偶尔接近边界时没有梯度引导，导致突然终止。
- **signal**: 缺少 height boundary precursor penalty，使得 agent 无法预判高度危险。
- **level**: Level 2
- **hypothesis**: 加入轻量 hinge 惩罚在高度接近 0.2 或 1.0 时提供“远离边界”的梯度，能在不完全牺牲速度的前提下降低终止率，从而提升平均得分并延长有效步长。
- **risk**: 若惩罚过强可能使 agent 过度保守，压低前进速度；当前系数 0.3 和较窄的预警区（边界外 0.05）旨在最小化副作用。若训练后 height_boundary_penalty active_rate 飙升或 len 下降，需削弱系数或拉宽边界。
