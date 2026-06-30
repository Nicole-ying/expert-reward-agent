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
    
    # 2. Landing-quality shaping (weakened coefficient, relaxed threshold)
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    near_target_factor = max(0.0, 1.0 - next_distance / 2.0)  # relaxed back to 2.0
    low_speed_factor = max(0.0, 1.0 - speed / 2.0)
    upright_factor = max(0.0, 1.0 - abs(next_body_angle) / 0.5)
    
    landing_quality = near_target_factor * low_speed_factor * upright_factor
    landing_shaping_reward = 0.5 * landing_quality  # weakened from 1.5 to 0.5
    
    # 3. Small distance anchor to encourage moving towards target (kept, unchanged)
    distance_reward = -0.1 * next_distance
    
    # Combine components
    total_reward = progress_delta_reward + landing_shaping_reward + distance_reward
    
    # Build components dict
    components = {
        "progress_delta_reward": progress_delta_reward,
        "landing_shaping_reward": landing_shaping_reward,
        "distance_reward": distance_reward,
        "total_reward": total_reward
    }
    
    return float(total_reward), components