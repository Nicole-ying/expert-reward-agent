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
    
    # --- Component 1: Progress Delta Reward (revert to best coefficient 50) ---
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_reward = 50.0 * progress_delta
    
    # --- Component 2: Stability Penalty (revert to best light weights) ---
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    angle_penalty = 0.2 * abs(next_body_angle)
    angular_vel_penalty = 0.1 * abs(next_angular_vel)
    speed_penalty = 0.15 * speed
    stability_penalty = -(angle_penalty + angular_vel_penalty + speed_penalty)
    
    # --- Component 3: Landing Shaping (revert to best relaxed thresholds and coefficient) ---
    near_target = max(0.0, 1.0 - next_dist / 1.0)
    low_speed = max(0.0, 1.0 - speed / 1.0)
    stable_angle = max(0.0, 1.0 - abs(next_body_angle) / 0.5)
    both_contact = 1.0 if (next_left_contact > 0.5 and next_right_contact > 0.5) else 0.0
    landing_shaping = 3.0 * near_target * low_speed * stable_angle + 2.0 * both_contact * near_target * low_speed
    
    # --- Total Reward ---
    total_reward = progress_reward + stability_penalty + landing_shaping
    
    # --- Components dict ---
    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "landing_shaping": landing_shaping,
        "total_reward": total_reward
    }
    
    return float(total_reward), components