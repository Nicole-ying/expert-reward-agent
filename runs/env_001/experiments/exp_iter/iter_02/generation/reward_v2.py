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
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_reward = 10.0 * progress_delta
    
    # 2. Smoother landing-quality shaping (revised from soft_landing_bonus)
    # Use a continuous bonus that increases as the agent gets closer to the target,
    # moves slower, and maintains a more upright posture.
    # This replaces the hard sparse bonus with a smoother gradient.
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    landing_quality = 0.0
    # Distance component: reward being close to target
    distance_factor = max(0.0, 1.0 - next_dist / 0.5)  # 0 when far, 1 when at target
    landing_quality += 1.0 * distance_factor
    # Speed component: reward low speed
    speed_factor = max(0.0, 1.0 - speed / 0.5)  # 0 when fast, 1 when still
    landing_quality += 1.0 * speed_factor
    # Angle component: reward upright posture
    angle_factor = max(0.0, 1.0 - abs(next_body_angle) / 0.2)  # 0 when tilted, 1 when upright
    landing_quality += 1.0 * angle_factor
    # Contact component: small bonus for having both feet on the ground
    both_contact = (next_left_contact > 0.5) and (next_right_contact > 0.5)
    contact_bonus = 0.5 if both_contact else 0.0
    landing_quality += contact_bonus
    
    # Scale the landing quality to provide meaningful guidance without dominating
    landing_shaping_reward = 0.5 * landing_quality
    
    # 3. Stability penalty (kept but reduced weight to prevent dominance)
    angle_penalty = -0.3 * abs(next_body_angle)
    angular_vel_penalty = -0.1 * abs(next_angular_vel)
    speed_penalty = -0.2 * speed
    stability_penalty = angle_penalty + angular_vel_penalty + speed_penalty
    
    # Total reward
    total_reward = progress_reward + stability_penalty + landing_shaping_reward
    
    # Components dict
    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "landing_shaping_reward": landing_shaping_reward,
        "total_reward": total_reward
    }
    
    return float(total_reward), components