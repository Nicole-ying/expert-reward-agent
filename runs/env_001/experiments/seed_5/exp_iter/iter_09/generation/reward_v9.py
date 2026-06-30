def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取观测变量
    x_pos = obs[0]
    y_pos = obs[1]
    x_vel = obs[2]
    y_vel = obs[3]
    body_angle = obs[4]
    angular_vel = obs[5]
    left_contact = obs[6]
    right_contact = obs[7]
    
    next_x_pos = next_obs[0]
    next_y_pos = next_obs[1]
    next_x_vel = next_obs[2]
    next_y_vel = next_obs[3]
    next_body_angle = next_obs[4]
    next_angular_vel = next_obs[5]
    next_left_contact = next_obs[6]
    next_right_contact = next_obs[7]
    
    # 1. 主学习信号：progress_delta_reward (保留，增加训练进度缩放)
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_scale = 1.0 + 0.5 * training_progress
    progress_reward = 10.0 * progress_scale * progress_delta
    
    # 2. 稳定约束：stability_penalty (进一步削弱)
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    angle_penalty = -0.015 * abs(next_body_angle)
    angular_vel_penalty = -0.005 * abs(next_angular_vel)
    speed_penalty = -0.002 * speed
    stability_penalty = angle_penalty + angular_vel_penalty + speed_penalty
    
    # 3. 着陆质量塑造：landing_shaping_reward (保留权重不变)
    near_target = max(0.0, 1.0 - next_dist / 0.5)
    low_speed = max(0.0, 1.0 - speed / 0.3)
    stable_angle = max(0.0, 1.0 - abs(next_body_angle) / 0.2)
    both_contact = 1.0 if (next_left_contact > 0.5) and (next_right_contact > 0.5) else 0.0
    
    landing_shaping_reward = 0.5 * near_target + 0.25 * low_speed + 0.25 * stable_angle + 0.25 * both_contact
    
    # 4. 接近奖励：approach_reward (保留，权重不变)
    approach_reward = 2.0 * max(0.0, 1.0 - next_dist / 1.0)
    
    # 5. 小权重距离锚点：distance_anchor (保留)
    distance_anchor = -0.1 * next_dist
    
    # 组合总奖励
    total_reward = progress_reward + stability_penalty + landing_shaping_reward + approach_reward + distance_anchor
    
    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "landing_shaping_reward": landing_shaping_reward,
        "approach_reward": approach_reward,
        "distance_anchor": distance_anchor,
        "total_reward": total_reward
    }
    
    return float(total_reward), components