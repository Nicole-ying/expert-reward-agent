def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observations
    x_pos = obs[0]
    y_pos = obs[1]
    x_vel = obs[2]
    y_vel = obs[3]
    body_angle = obs[4]
    ang_vel = obs[5]
    left_contact = obs[6]
    right_contact = obs[7]
    
    next_x_pos = next_obs[0]
    next_y_pos = next_obs[1]
    next_x_vel = next_obs[2]
    next_y_vel = next_obs[3]
    next_body_angle = next_obs[4]
    next_ang_vel = next_obs[5]
    next_left_contact = next_obs[6]
    next_right_contact = next_obs[7]
    
    # 1. Main learning signal: progress_delta_reward (unchanged)
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_reward = 150.0 * progress_delta
    
    # 2. Reduced stability penalty (unchanged)
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    angle_penalty = 0.01 * abs(next_body_angle)
    speed_penalty = 0.01 * speed
    ang_vel_penalty = 0.005 * abs(next_ang_vel)
    stability_penalty = -(angle_penalty + speed_penalty + ang_vel_penalty)
    
    # 3. Smooth landing-quality shaping (unchanged)
    near_target = max(0.0, 1.0 - next_dist / 2.0)
    low_speed = max(0.0, 1.0 - speed / 1.0)
    stable_angle = max(0.0, 1.0 - abs(next_body_angle) / 0.5)
    landing_shaping = 1.0 * near_target + 1.0 * low_speed + 1.0 * stable_angle
    
    # 4. Distance reward - strengthened to provide stronger pull toward target
    distance_reward = -0.5 * next_dist  # increased from -0.1
    
    # Combine components
    total_reward = progress_reward + stability_penalty + landing_shaping + distance_reward
    
    # Build components dict
    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "landing_shaping": landing_shaping,
        "distance_reward": distance_reward,
        "total_reward": total_reward
    }
    
    return float(total_reward), components