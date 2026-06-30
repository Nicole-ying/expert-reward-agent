def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract relevant observations
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
    
    # Component 1: Progress delta reward (main learning signal)
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_reward = progress_delta * 10.0
    
    # Component 2: Distance anchor (small continuous guidance)
    # Provides a small penalty for being far from target, ensuring baseline signal
    distance_anchor = -0.1 * next_dist
    
    # Component 3: Stability penalty (weakened to avoid dominating)
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    angle_penalty = abs(next_body_angle) * 0.2
    angular_vel_penalty = abs(next_angular_vel) * 0.1
    speed_penalty = speed * 0.1
    stability_penalty = -(angle_penalty + angular_vel_penalty + speed_penalty)
    
    # Component 4: Smooth landing shaping (replaces sparse soft_landing_bonus)
    # Continuous quality score that activates gradually when near target
    # Uses sigmoid-like scaling to provide smooth gradient
    dist_factor = 1.0 / (1.0 + 5.0 * next_dist)  # ~1.0 when dist=0, ~0.17 when dist=1.0
    speed_quality = 1.0 / (1.0 + 2.0 * speed)     # ~1.0 when speed=0, ~0.33 when speed=1.0
    angle_quality = 1.0 / (1.0 + 5.0 * abs(next_body_angle))  # ~1.0 when angle=0
    contact_bonus = 0.0
    if next_left_contact > 0.5 and next_right_contact > 0.5:
        contact_bonus = 0.5
    
    landing_shaping = 2.0 * dist_factor * speed_quality * angle_quality + contact_bonus
    
    # Component 5: Small action penalty (efficiency)
    action_penalty = 0.0
    if action != 0:
        action_penalty = -0.05
    
    # Combine components
    total_reward = progress_reward + distance_anchor + stability_penalty + landing_shaping + action_penalty
    
    # Build components dict
    components = {
        "progress_reward": progress_reward,
        "distance_anchor": distance_anchor,
        "stability_penalty": stability_penalty,
        "landing_shaping": landing_shaping,
        "action_penalty": action_penalty,
        "total_reward": total_reward
    }
    
    return float(total_reward), components