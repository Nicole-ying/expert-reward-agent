def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：奖励沿前进方向的速度
    w_forward = 1.0
    forward_velocity = next_obs[13]  # 前进速度（主体x方向速度）
    forward_reward = w_forward * forward_velocity

    # 约束1：惩罚横向运动，鼓励直线前进
    w_lateral = 0.5
    lateral_velocity = next_obs[14]  # 横向速度
    lateral_penalty = -w_lateral * (lateral_velocity ** 2)

    # 约束2：惩罚过高的角速度，保持姿态稳定
    w_angular = 0.5
    roll_rate = next_obs[16]
    pitch_rate = next_obs[17]
    yaw_rate = next_obs[18]
    angular_penalty = -w_angular * (roll_rate ** 2 + pitch_rate ** 2 + yaw_rate ** 2)

    # 约束3：惩罚剧烈的垂直运动，防止跳跃/坠落
    w_vertical = 0.1
    vertical_velocity = next_obs[15]  # 垂直速度
    vertical_penalty = -w_vertical * abs(vertical_velocity)

    total_reward = forward_reward + lateral_penalty + angular_penalty + vertical_penalty

    components = {
        "forward_velocity_reward": forward_reward,
        "lateral_velocity_penalty": lateral_penalty,
        "angular_velocity_penalty": angular_penalty,
        "vertical_velocity_penalty": vertical_penalty,
    }

    return float(total_reward), components