def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    body_z = next_obs[0]
    quat_x = next_obs[2]
    quat_y = next_obs[3]
    body_x_vel = next_obs[13]

    # 直立因子
    body_up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)
    upright_factor = max(0.0, body_up_z)

    # 高度安全 gate
    low_safe = 0.35
    min_z = 0.2
    high_safe = 0.9
    max_z = 1.0

    if body_z >= low_safe:
        gate_low = 1.0
    else:
        gate_low = max(0.0, (body_z - min_z) / (low_safe - min_z))

    if body_z <= high_safe:
        gate_high = 1.0
    else:
        gate_high = max(0.0, (max_z - body_z) / (max_z - high_safe))

    height_gate = min(gate_low, gate_high)

    # 前进奖励（乘以上述两个 gate）
    w_forward = 1.0
    forward = w_forward * body_x_vel * upright_factor * height_gate

    # 直立偏差惩罚（保持上一轮的系数）
    upright_pen = -2.0 * (quat_x**2 + quat_y**2)

    total_reward = forward + upright_pen

    components = {
        "forward": forward,
        "upright_pen": upright_pen,
        "height_gate": height_gate
    }
    return float(total_reward), components