# Response Record

# 设计理由
当前 agent 学会了长时间存活（len≈797），但前向速度极低，导致主信号 `forward_upright_gated` 每步仅约 0.18，无法抗衡惩罚项。因此总奖励近似持平，甚至因外部环境评分机制使最终得分极负。问题根源是前进奖励的梯度太弱，无法驱动 agent 放弃“站立存活”的低回报策略。本次采用 **Level 1 尺度修复**：将前进速度的系数从 1.0 上调至 **10.0**，使高速前进能够产生压倒性奖励，迫使 agent 走出局部最优。其他组件（高度门、横向/旋转/力矩惩罚）保持不变，因为它们已经在有效约束而未被过度惩罚。  
设计校准：主信号 per‑step 预期可达 ~3.0（速度 0.5‑0.6 时），惩罚合计 ~0.07，负担远低于 0.5× 上限，安全。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract key signals from observation (current obs, not next_obs)
    body_z = obs[0]
    qx = obs[2]
    qy = obs[3]
    body_x_vel = obs[13]
    body_y_vel = obs[14]
    body_roll_vel = obs[16]
    body_pitch_vel = obs[17]
    body_yaw_vel = obs[18]

    # ----- uprightness (projection of body z-axis onto world z-axis) -----
    body_up_z = 1.0 - 2.0 * (qx**2 + qy**2)
    body_up_z = max(0.0, min(1.0, body_up_z))  # numerical safety clamp

    # ----- core learning signal: forward velocity modulated by uprightness -----
    # Scale up drastically to escape the low‑speed local optimum.
    forward_reward = 10.0 * body_x_vel * body_up_z

    # ----- height survival gate (continuous, bounded [0, 1]) -----
    z_low_safe = 0.4
    z_low_danger = 0.3
    z_high_safe = 0.8
    z_high_danger = 0.9

    if body_z >= z_low_danger and body_z <= z_low_safe:
        height_gate = (body_z - z_low_danger) / (z_low_safe - z_low_danger)
    elif body_z >= z_low_safe and body_z <= z_high_safe:
        height_gate = 1.0
    elif body_z >= z_high_safe and body_z <= z_high_danger:
        height_gate = (z_high_danger - body_z) / (z_high_danger - z_high_safe)
    else:
        height_gate = 0.0

    # Apply gate to forward reward — bad height suppresses all forward credit
    gated_forward = forward_reward * height_gate

    # ----- lateral stability penalty (hinge on |body_y_velocity|) -----
    lateral_threshold = 0.15
    lateral_error = max(0.0, abs(body_y_vel) - lateral_threshold)
    w_lateral = 0.1
    lateral_penalty = -w_lateral * lateral_error

    # ----- rotation stability penalty (hinge on angular velocities) -----
    rotation_threshold = 0.5
    roll_error = max(0.0, abs(body_roll_vel) - rotation_threshold)
    pitch_error = max(0.0, abs(body_pitch_vel) - rotation_threshold)
    yaw_error = max(0.0, abs(body_yaw_vel) - rotation_threshold)
    w_rotation = 0.05
    rotation_penalty = -w_rotation * (roll_error + pitch_error + yaw_error)

    # ----- action penalty for energy efficiency (unchanged) -----
    w_action = 0.03
    torque_penalty = -w_action * sum(action**2) / 8.0

    total_reward = gated_forward + lateral_penalty + rotation_penalty + torque_penalty

    components = {
        "forward_upright_gated": gated_forward,
        "height_gate_value": height_gate,
        "lateral_penalty": lateral_penalty,
        "rotation_penalty": rotation_penalty,
        "torque_penalty": torque_penalty
    }

    return float(total_reward), components
```

# 诊断摘要
- **audit**: 前进速度信号太弱（每步 ~0.18），无法推动 agent 脱离低速存活策略；缺少对更高速度的强烈激励。
- **behavior**: agent 学会了在安全高度内长时间站立/缓慢移动，利用微小前向速度换取微弱正奖励，导致总得分暴降。
- **signal**: 缺乏足够强的速度梯度，现有的线性系数（1.0）在低速度区提供的进步动力不足。
- **level**: Level 1
- **hypothesis**: 将前进奖励系数放大 10 倍后，高速行走的回报将显著超过苟活策略，agent 会被迫学习更快、更稳定的步态，从而大幅提高前进距离得分。
- **risk**: 若奖励过大，agent 可能牺牲平衡追求瞬时高速，导致摔倒终止增加，但后续若摔倒频繁可通过微调系数解决。
