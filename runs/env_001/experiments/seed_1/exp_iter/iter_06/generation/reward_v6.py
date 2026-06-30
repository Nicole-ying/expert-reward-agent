def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observations
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
    
    # 1. Main learning signal: progress_delta_reward (kept, unchanged)
    current_distance = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_distance = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_delta = current_distance - next_distance
    progress_delta_reward = 10.0 * progress_delta
    
    # 2. Stability penalty (kept, unchanged from last iteration)
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    angle_penalty = 0.1 * abs(next_body_angle)
    angular_vel_penalty = 0.05 * abs(next_angular_vel)
    speed_penalty = 0.1 * speed
    stability_penalty = -(angle_penalty + angular_vel_penalty + speed_penalty)
    
    # 3. Simplified landing-quality shaping (kept, unchanged from last iteration)
    near_target_factor = max(0.0, 1.0 - next_distance / 2.0)
    low_speed_factor = max(0.0, 1.0 - speed / 2.0)
    upright_factor = max(0.0, 1.0 - abs(next_body_angle) / 0.5)
    
    landing_quality = near_target_factor * low_speed_factor * upright_factor
    landing_shaping_reward = 1.0 * landing_quality
    
    # 4. Small distance anchor to prevent getting stuck far away (kept, unchanged from last iteration)
    distance_anchor = -0.05 * next_distance
    
    # Combine components
    total_reward = progress_delta_reward + stability_penalty + landing_shaping_reward + distance_anchor
    
    # Build components dict
    components = {
        "progress_delta_reward": progress_delta_reward,
        "stability_penalty": stability_penalty,
        "landing_shaping_reward": landing_shaping_reward,
        "distance_anchor": distance_anchor,
        "total_reward": total_reward
    }
    
    return float(total_reward), components