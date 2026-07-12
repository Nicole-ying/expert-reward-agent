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

    # 前进奖励（不再使用高度 gate，保证任何存活状态下都有梯度）
    base_forward = max(0.0, vx)
    forward_reward = 1.0 * base_forward * posture_factor

    # 侧向速度惩罚
    lateral_penalty = -0.5 * (vy ** 2)

    # 动作幅度惩罚
    action_penalty = -0.1 * sum(a ** 2 for a in action)

    # 高度惩罚：低于 0.35 m 开始线性增大，最大惩罚量级受控
    height_penalty = -0.5 * max(0.0, 0.35 - body_z)

    # 下降速度惩罚：阻止过快坠落
    descend_penalty = -0.2 * max(0.0, -0.5 - vz)

    total_reward = forward_reward + lateral_penalty + action_penalty + height_penalty + descend_penalty

    components = {
        "forward_reward": forward_reward,
        "lateral_penalty": lateral_penalty,
        "action_penalty": action_penalty,
        "height_penalty": height_penalty,
        "descend_penalty": descend_penalty
    }
    return float(total_reward), components