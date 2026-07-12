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