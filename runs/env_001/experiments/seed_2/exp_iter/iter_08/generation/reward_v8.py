def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取观测变量
    x_pos = obs[0]
    y_pos = obs[1]
    next_x_pos = next_obs[0]
    next_y_pos = next_obs[1]
    next_x_vel = next_obs[2]
    next_y_vel = next_obs[3]
    next_body_angle = next_obs[4]
    next_left_contact = next_obs[6]
    next_right_contact = next_obs[7]

    # 1. 主学习信号：progress_delta_reward (保留，权重不变)
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_reward = 10.0 * progress_delta

    # 2. 平滑接近与着陆引导：smooth_landing_shaping (微调权重)
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    near_target_factor = max(0.0, 1.0 - next_dist / 2.0)
    speed_comfort = max(0.0, 1.0 - speed / 0.8)
    angle_comfort = max(0.0, 1.0 - abs(next_body_angle) / 0.3)
    smooth_landing_shaping = 1.2 * near_target_factor * (0.5 * speed_comfort + 0.5 * angle_comfort)

    # 3. 接触着陆奖励：contact_landing_bonus (保留)
    contact = (next_left_contact > 0.5) or (next_right_contact > 0.5)
    near_target = next_dist < 0.8
    low_speed = speed < 0.5
    stable_angle = abs(next_body_angle) < 0.2
    if contact and near_target and low_speed and stable_angle:
        contact_bonus = 1.0
    else:
        contact_bonus = 0.0

    # 4. 距离锚点：distance_reward (保留)
    distance_reward = -0.1 * next_dist

    # 组合总奖励
    total_reward = progress_reward + smooth_landing_shaping + contact_bonus + distance_reward

    components = {
        "progress_reward": progress_reward,
        "smooth_landing_shaping": smooth_landing_shaping,
        "contact_landing_bonus": contact_bonus,
        "distance_reward": distance_reward,
        "total_reward": total_reward
    }

    return float(total_reward), components