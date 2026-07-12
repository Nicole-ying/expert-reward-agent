def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    body_z = next_obs[0]
    quat_x = next_obs[2]
    quat_y = next_obs[3]
    quat_z = next_obs[4]
    vx = next_obs[13]
    vy = next_obs[14]
    vz = next_obs[15]

    # 姿态健康因子（连续，保持直立）
    pose_error = quat_x**2 + quat_y**2 + quat_z**2
    posture_factor = 2.718281828 ** (-10.0 * pose_error)

    # 高度健康 gate（上、下界前兆）
    lower_safe = max(0.0, min((body_z - 0.2) / 0.1, 1.0))   # 0.2→0, 0.3→1
    upper_safe = max(0.0, min((1.0 - body_z) / 0.1, 1.0))   # 1.0→0, 0.9→1
    height_factor = lower_safe * upper_safe

    # 前进奖励，同时受姿态和高度 gate 调制
    base_forward = max(0.0, vx)
    forward_reward = 1.0 * base_forward * posture_factor * height_factor

    # 侧向速度惩罚
    lateral_penalty = -0.5 * (vy ** 2)

    # 下降速度惩罚：阻止过快坠落
    descend_penalty = -0.2 * max(0.0, -0.5 - vz)

    total_reward = forward_reward + lateral_penalty + descend_penalty

    components = {
        "forward_reward": forward_reward,
        "lateral_penalty": lateral_penalty,
        "descend_penalty": descend_penalty
    }
    return float(total_reward), components