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
    distance_anchor = -0.1 * next_dist
    
    # Component 3: Stability penalty (weakened to avoid dominating)
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    angle_penalty = abs(next_body_angle) * 0.1
    angular_vel_penalty = abs(next_angular_vel) * 0.05
    speed_penalty = speed * 0.05
    stability_penalty = -(angle_penalty + angular_vel_penalty + speed_penalty)
    
    # Component 4: Proximity reward (smooth approach guidance, replaces landing_shaping)
    # Provides a continuous reward that increases as the agent gets closer to the target.
    # This is a simpler and more direct signal than the previous product-based shaping.
    proximity_reward = 1.0 / (1.0 + 5.0 * next_dist)
    
    # Component 5: Small action penalty (efficiency)
    action_penalty = 0.0
    if action != 0:
        action_penalty = -0.05
    
    # Combine components
    total_reward = progress_reward + distance_anchor + stability_penalty + proximity_reward + action_penalty
    
    # Build components dict
    components = {
        "progress_reward": progress_reward,
        "distance_anchor": distance_anchor,
        "stability_penalty": stability_penalty,
        "proximity_reward": proximity_reward,
        "action_penalty": action_penalty,
        "total_reward": total_reward
    }
    
    return float(total_reward), components