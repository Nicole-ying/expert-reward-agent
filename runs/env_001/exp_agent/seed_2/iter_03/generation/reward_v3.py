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
    
    # 1. Main learning signal: progress_delta_reward
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_delta_reward = 10.0 * progress_delta
    
    # 2. Stability penalty - moderate strength, no distance gate
    # The agent dies early because it falls over while moving toward target
    # Need enough penalty to teach upright posture
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    angle_penalty = -0.1 * abs(next_body_angle)
    angular_vel_penalty = -0.05 * abs(next_angular_vel)
    speed_penalty = -0.03 * speed
    stability_penalty = angle_penalty + angular_vel_penalty + speed_penalty
    
    # 3. Soft landing proxy: continuous product of bounded factors
    # Make contact_factor continuous too (use raw contact values)
    proximity_factor = 1.0 / (1.0 + 5.0 * next_dist)
    speed_factor = 1.0 / (1.0 + 5.0 * speed)
    angle_factor = 1.0 / (1.0 + 10.0 * abs(next_body_angle))
    angular_vel_factor = 1.0 / (1.0 + 5.0 * abs(next_angular_vel))
    # Continuous contact factor - use raw values as probabilities
    contact_factor = next_left_contact * next_right_contact  # continuous in [0,1]
    
    soft_landing_bonus = 10.0 * proximity_factor * speed_factor * angle_factor * angular_vel_factor * contact_factor
    
    # 4. Small energy penalty for using engines
    energy_penalty = 0.0
    if action != 0:
        energy_penalty = -0.05
    
    total_reward = progress_delta_reward + stability_penalty + soft_landing_bonus + energy_penalty
    
    components = {
        "progress_delta_reward": progress_delta_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_bonus": soft_landing_bonus,
        "energy_penalty": energy_penalty,
        "total_reward": total_reward
    }
    
    return float(total_reward), components