def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 距离进展
    x_curr = obs[0]
    y_curr = obs[1]
    dist_curr = (x_curr ** 2 + y_curr ** 2) ** 0.5

    x_next = next_obs[0]
    y_next = next_obs[1]
    dist_next = (x_next ** 2 + y_next ** 2) ** 0.5

    progress = dist_curr - dist_next

    # 姿态与角速度惩罚（保持低摇晃）
    body_angle_next = next_obs[4]
    angular_vel_next = next_obs[5]
    angle_penalty = body_angle_next ** 2 + angular_vel_next ** 2

    # 连续软着陆引导
    dist_factor = 2.718281828 ** (-dist_next / 0.5)
    x_vel_next = next_obs[2]
    y_vel_next = next_obs[3]
    speed_factor = max(0.0, 1.0 - (abs(x_vel_next) + abs(y_vel_next)) / 1.0)
    landing_reward = dist_factor * speed_factor

    # 新增：双腿接触完成奖励
    contact_both = next_obs[6] * next_obs[7]   # 0 或 1
    contact_reward = contact_both * 5.0

    total = (
        10.0 * progress
        - 0.5 * angle_penalty
        + 0.01 * landing_reward
        + contact_reward
    )

    components = {
        "progress": 10.0 * progress,
        "angle_penalty": -0.5 * angle_penalty,
        "landing_reward": 0.01 * landing_reward,
        "contact_reward": contact_reward
    }

    return float(total), components