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
    # Distance to goal (target is at origin, obs are relative coordinates)
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_scale = 5.0
    progress_reward = progress_scale * progress_delta
    
    # 2. Stability penalty: penalize high velocity, large angle, and angular velocity
    # Use next_obs to penalize the resulting state after action
    vel_magnitude = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    angle_penalty_weight = 0.5
    vel_penalty_weight = 0.3
    angular_vel_penalty_weight = 0.2
    
    # Angle penalty: penalize deviation from upright (angle=0)
    angle_penalty = -angle_penalty_weight * abs(next_body_angle)
    
    # Velocity penalty: penalize high speed (encourage slow approach)
    vel_penalty = -vel_penalty_weight * vel_magnitude
    
    # Angular velocity penalty: penalize spinning
    angular_vel_penalty = -angular_vel_penalty_weight * abs(next_angular_vel)
    
    stability_penalty = angle_penalty + vel_penalty + angular_vel_penalty
    
    # 3. Soft landing proxy: small bonus when near target, slow, stable, and both contacts
    near_target_threshold = 0.3
    low_speed_threshold = 0.5
    stable_angle_threshold = 0.2
    both_contact = (next_left_contact > 0.5) and (next_right_contact > 0.5)
    
    near_target = next_dist < near_target_threshold
    low_speed = vel_magnitude < low_speed_threshold
    stable_angle = abs(next_body_angle) < stable_angle_threshold
    
    soft_landing_bonus = 0.0
    if near_target and low_speed and stable_angle and both_contact:
        soft_landing_bonus = 2.0
    
    # 4. Small action penalty to encourage fuel efficiency (very small weight)
    # action 0 = no engine, action 1/2/3 = engine use
    action_penalty_weight = 0.05
    action_penalty = 0.0
    if action != 0:  # Any engine use
        action_penalty = -action_penalty_weight
    
    # Combine all components
    total_reward = progress_reward + stability_penalty + soft_landing_bonus + action_penalty
    
    # Build components dict
    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_bonus": soft_landing_bonus,
        "action_penalty": action_penalty,
        "total_reward": total_reward
    }
    
    return float(total_reward), components