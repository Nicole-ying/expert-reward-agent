def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：前进速度 + 稳定性门控
    hull_angle = obs[0]                  # 身体倾斜角
    horizontal_velocity = obs[2]         # 前向速度
    k_angle = 1.0
    stability = 1.0 / (1.0 + k_angle * abs(hull_angle))
    forward_vel = max(horizontal_velocity, 0.0)
    w_progress = 1.0
    forward_progress = w_progress * stability * forward_vel

    # 姿态惩罚：鼓励保持直立
    w_posture = 0.1
    posture_penalty = -w_posture * (hull_angle ** 2)

    # 角速度惩罚：抑制剧烈旋转
    hull_angular_velocity = obs[1]
    w_angvel = 0.01
    angvel_penalty = -w_angvel * (hull_angular_velocity ** 2)

    # 动作效率惩罚：极小权重降低能耗
    w_action = 0.001
    action_sum_sq = sum(a**2 for a in action)
    action_penalty = -w_action * action_sum_sq

    total_reward = forward_progress + posture_penalty + angvel_penalty + action_penalty

    components = {
        "forward_progress": forward_progress,
        "posture_penalty": posture_penalty,
        "angular_vel_penalty": angvel_penalty,
        "action_penalty": action_penalty
    }
    return float(total_reward), components