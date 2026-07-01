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
    
    # --- Component 1: Progress Delta Reward (main learning signal) ---
    # Distance to goal (origin) at current step
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    # Distance to goal at next step
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    # Progress: positive when moving closer to goal
    progress_delta = current_dist - next_dist
    progress_reward = 10.0 * progress_delta
    
    # --- Component 2: Stability Penalty (light constraint) ---
    # Penalize high speed, large angle, and high angular velocity
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    angle_penalty = 0.5 * abs(next_body_angle)
    angular_vel_penalty = 0.2 * abs(next_angular_vel)
    speed_penalty = 0.3 * speed
    stability_penalty = -(angle_penalty + angular_vel_penalty + speed_penalty)
    
    # --- Component 3: Soft Landing Proxy (small bonus for good landing conditions) ---
    # Conditions: near target, low speed, stable angle, both supports contacting
    near_target = next_dist < 0.5
    low_speed = speed < 0.5
    stable_angle = abs(next_body_angle) < 0.3
    both_contact = (next_left_contact > 0.5) and (next_right_contact > 0.5)
    
    soft_landing_bonus = 0.0
    if near_target and low_speed and stable_angle and both_contact:
        soft_landing_bonus = 2.0
    
    # --- Total Reward ---
    total_reward = progress_reward + stability_penalty + soft_landing_bonus
    
    # --- Components dict ---
    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_bonus": soft_landing_bonus,
        "total_reward": total_reward
    }
    
    return float(total_reward), components