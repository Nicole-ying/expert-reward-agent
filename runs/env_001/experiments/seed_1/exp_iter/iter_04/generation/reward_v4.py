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
    
    # 1. Main learning signal: progress_delta_reward (kept)
    current_distance = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_distance = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_delta = current_distance - next_distance
    progress_delta_reward = 10.0 * progress_delta
    
    # 2. Stability penalty: penalize high velocity, large angle, and high angular velocity (kept)
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    angle_penalty = 0.3 * abs(next_body_angle)
    angular_vel_penalty = 0.1 * abs(next_angular_vel)
    speed_penalty = 0.2 * speed
    stability_penalty = -(angle_penalty + angular_vel_penalty + speed_penalty)
    
    # 3. Smoother landing-quality shaping (revised: increased weight for stronger guidance)
    near_target_factor = max(0.0, 1.0 - next_distance / 2.0)
    low_speed_factor = max(0.0, 1.0 - speed / 2.0)
    upright_factor = max(0.0, 1.0 - abs(next_body_angle) / 0.5)
    
    both_contact = (next_left_contact > 0.5) and (next_right_contact > 0.5)
    contact_factor = 1.0 + 0.5 * both_contact
    
    landing_quality = near_target_factor * low_speed_factor * upright_factor * contact_factor
    landing_shaping_reward = 3.0 * landing_quality  # increased from 2.0 to 3.0
    
    # 4. Small distance anchor to prevent getting stuck far away (kept, slightly increased)
    distance_anchor = -0.2 * next_distance  # increased from -0.1 to -0.2
    
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