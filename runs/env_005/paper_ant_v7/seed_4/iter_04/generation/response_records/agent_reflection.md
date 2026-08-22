# Response Record

# 设计理由
本轮为 **Level 1 尺度修复**：责任组件齐全，但惩罚项的强度远超主信号，导致 agent 几乎无法存活。训练反馈显示 `rotation_penalty` 每步均值约 -0.42，`lateral_penalty` 约 -0.20，而主信号 `forward_upright_gated` 每步只有 0.59。两个惩罚的 per-step 负担合计 ≈ -0.62，超过了主信号，违背“所有惩罚合计 ≤ 主信号 0.5 倍”的校准原则。这直接导致 episode 长度从 349 暴跌至 16，agent 快速倒下并终止，没有机会学习前进步态。

干预方式：大幅降低旋转和侧向惩罚的系数，使每步总惩罚负荷降至主信号的 0.3 倍以内。其他组件（gate、前进奖励、动作惩罚）保持不变。若 agent 能存活更久，它就有更多梯度信号去改善直立和前向运动，从而逐步减少旋转和侧向误差。

- `w_rotation` 从 0.2 降为 **0.05**（保留 0.5 rad/s 阈值）。
- `w_lateral` 从 0.5 降为 **0.1**（保留 0.15 m/s 阈值）。
- `w_action` 保持 0.03（其负担极小，不影响整体平衡）。

预期效果：每步惩罚负担降至约 -0.15，agent 存活步数应回升至数百步，score 逐渐推高。

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
    forward_reward = body_x_vel * body_up_z

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
    w_lateral = 0.1  # reduced from 0.5
    lateral_penalty = -w_lateral * lateral_error

    # ----- rotation stability penalty (hinge on angular velocities) -----
    rotation_threshold = 0.5
    roll_error = max(0.0, abs(body_roll_vel) - rotation_threshold)
    pitch_error = max(0.0, abs(body_pitch_vel) - rotation_threshold)
    yaw_error = max(0.0, abs(body_yaw_vel) - rotation_threshold)
    w_rotation = 0.05  # reduced from 0.2
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
- **audit**: 惩罚负担超过主信号，致使 agent 存活步数极短，高度 gate 虽覆盖终止条件但强度没有问题。
- **behavior**: agent 几乎立刻终止（len≈16），无法学习行走。
- **signal**: 旋转惩罚和侧向惩罚系数过大，压倒了正向前进信号。
- **level**: Level 1
- **hypothesis**: 减小旋转和侧向惩罚系数使每步净奖励为正，agent 存活时间变长后能获取更多前进奖励，从而自发降低角速度。
- **risk**: 若 agent 在存活变长后仍不能学会直立行走，可能需进一步降低 gate 的安全阈值或提高前进奖励，但本轮仅调惩罚系数，副作用小。
