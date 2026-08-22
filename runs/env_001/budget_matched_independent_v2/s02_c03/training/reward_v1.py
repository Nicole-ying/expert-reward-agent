def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract signals from next_obs
    x_pos = next_obs[0]      # horizontal coordinate relative to target pad
    y_pos = next_obs[1]      # vertical coordinate relative to pad height
    x_vel = next_obs[2]      # horizontal linear velocity
    y_vel = next_obs[3]      # vertical linear velocity
    body_angle = next_obs[4] # orientation angle
    ang_vel = next_obs[5]    # angular velocity
    left_contact = next_obs[6]  # left support contact flag
    right_contact = next_obs[7] # right support contact flag
    
    # Also extract from obs for delta calculations
    prev_x_pos = obs[0]
    prev_y_pos = obs[1]
    prev_x_vel = obs[2]
    prev_y_vel = obs[3]
    prev_body_angle = obs[4]
    prev_ang_vel = obs[5]
    
    # Distance to target (target is at origin)
    distance = (x_pos ** 2 + y_pos ** 2) ** 0.5
    
    # Speed magnitude
    speed = (x_vel ** 2 + y_vel ** 2) ** 0.5
    
    # Contact flags
    any_contact = 1.0 if (left_contact > 0.5 or right_contact > 0.5) else 0.0
    both_contact = 1.0 if (left_contact > 0.5 and right_contact > 0.5) else 0.0
    
    # Reward for being close to target (exponential shaping)
    distance_reward = 2.718281828 ** (-2.0 * distance)
    
    # Reward for low speed when near target (encourages settling)
    speed_penalty = -0.1 * speed * (1.0 / (1.0 + 2.718281828 ** (-5.0 * (distance - 0.5))))
    
    # Reward for stable orientation (upright = angle near 0)
    angle_penalty = -0.05 * (body_angle ** 2)
    
    # Reward for low angular velocity
    ang_vel_penalty = -0.02 * (ang_vel ** 2)
    
    # Reward for having both contacts (settled on pad)
    contact_reward = 1.0 * both_contact
    
    # Reward for reducing distance over time (progress)
    prev_distance = (prev_x_pos ** 2 + prev_y_pos ** 2) ** 0.5
    progress_reward = 2.0 * max(0.0, prev_distance - distance)
    
    # Penalty for using main engine (action 2) to encourage fuel efficiency
    engine_penalty = -0.05 if action == 2 else 0.0
    
    # Small survival bonus to encourage exploration
    survival_bonus = 0.01
    
    # Combine rewards
    total_reward = (
        distance_reward +
        speed_penalty +
        angle_penalty +
        ang_vel_penalty +
        contact_reward +
        progress_reward +
        engine_penalty +
        survival_bonus
    )
    
    components = {
        "distance_reward": distance_reward,
        "speed_penalty": speed_penalty,
        "angle_penalty": angle_penalty,
        "ang_vel_penalty": ang_vel_penalty,
        "contact_reward": contact_reward,
        "progress_reward": progress_reward,
        "engine_penalty": engine_penalty,
        "survival_bonus": survival_bonus,
    }
    
    return float(total_reward), components