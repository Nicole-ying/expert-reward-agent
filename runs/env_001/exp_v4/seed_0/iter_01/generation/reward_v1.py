def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract state variables from observations
    # Current position (relative to target)
    x_pos = obs[0]
    y_pos = obs[1]
    
    # Next position (relative to target)
    next_x_pos = next_obs[0]
    next_y_pos = next_obs[1]
    
    # Current velocities
    x_vel = obs[2]
    y_vel = obs[3]
    
    # Next velocities
    next_x_vel = next_obs[2]
    next_y_vel = next_obs[3]
    
    # Orientation and angular velocity
    body_angle = obs[4]
    next_body_angle = next_obs[4]
    angular_vel = obs[5]
    next_angular_vel = next_obs[5]
    
    # Contact flags
    left_contact = obs[6]
    right_contact = obs[7]
    next_left_contact = next_obs[6]
    next_right_contact = next_obs[7]
    
    # ========== Component 1: Progress Delta Reward (Main Learning Signal) ==========
    # Reward the agent for moving closer to the target (0,0)
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    
    # Scale progress reward to be meaningful
    progress_scale = 5.0
    progress_reward = progress_delta * progress_scale
    
    # ========== Component 2: Stability Penalty (Light Constraint) ==========
    # Penalize high speed, large angle, and high angular velocity
    # Use next_obs to penalize the resulting state after action
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    angle_penalty_weight = 0.1
    speed_penalty_weight = 0.05
    angular_vel_penalty_weight = 0.02
    
    # Penalize deviation from upright (angle=0) and high angular velocity
    angle_penalty = -angle_penalty_weight * (next_body_angle ** 2)
    speed_penalty = -speed_penalty_weight * speed
    angular_vel_penalty = -angular_vel_penalty_weight * (next_angular_vel ** 2)
    
    stability_penalty = angle_penalty + speed_penalty + angular_vel_penalty
    
    # ========== Component 3: Soft Landing Proxy (Small Bonus) ==========
    # Small bonus when near target, low speed, stable angle, and both supports contact
    near_target_threshold = 0.3
    low_speed_threshold = 0.5
    stable_angle_threshold = 0.2
    
    is_near_target = next_dist < near_target_threshold
    is_low_speed = speed < low_speed_threshold
    is_stable_angle = abs(next_body_angle) < stable_angle_threshold
    is_both_contact = (next_left_contact > 0.5) and (next_right_contact > 0.5)
    
    landing_bonus_weight = 2.0
    if is_near_target and is_low_speed and is_stable_angle and is_both_contact:
        landing_bonus = landing_bonus_weight
    else:
        landing_bonus = 0.0
    
    # ========== Component 4: Small Action Penalty (Efficiency) ==========
    # Very light penalty for using engines to encourage fuel efficiency
    # action 0 = no engine, actions 1-3 = engine use
    action_penalty_weight = 0.02
    if action == 0:
        action_penalty = 0.0
    else:
        action_penalty = -action_penalty_weight
    
    # ========== Combine Components ==========
    total_reward = progress_reward + stability_penalty + landing_bonus + action_penalty
    
    # ========== Build Components Dictionary ==========
    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "angle_penalty": angle_penalty,
        "speed_penalty": speed_penalty,
        "angular_vel_penalty": angular_vel_penalty,
        "landing_bonus": landing_bonus,
        "action_penalty": action_penalty,
        "total_reward": total_reward
    }
    
    return float(total_reward), components