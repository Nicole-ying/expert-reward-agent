def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract state variables
    # Position (relative to target)
    x_pos = obs[0]
    y_pos = obs[1]
    next_x_pos = next_obs[0]
    next_y_pos = next_obs[1]
    
    # Velocity
    x_vel = next_obs[2]
    y_vel = next_obs[3]
    
    # Orientation
    body_angle = next_obs[4]
    angular_vel = next_obs[5]
    
    # Contact flags
    left_contact = next_obs[6]
    right_contact = next_obs[7]
    
    # ========== Component 1: Progress Delta Reward (main learning signal) ==========
    # Distance to target at current step
    dist_current = (x_pos ** 2 + y_pos ** 2) ** 0.5
    # Distance to target at next step
    dist_next = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    # Progress: positive when moving closer to target
    progress_delta = dist_current - dist_next
    # Strong coefficient to drive learning
    progress_reward = 50.0 * progress_delta
    
    # ========== Component 2: Distance Reward (small anchor, provides continuous gradient) ==========
    # Negative distance to target, scaled down to avoid dominance
    distance_reward = -0.1 * dist_next
    
    # ========== Component 3: Conditional Stability Penalty (only near target) ==========
    # Compute speed and angle
    speed = (x_vel ** 2 + y_vel ** 2) ** 0.5
    # Only penalize instability when close to target (dist_next < 0.5)
    # This allows high speed when far away, but requires stable approach near goal
    near_target_threshold = 0.5
    if dist_next < near_target_threshold:
        # Progressive penalty that increases as distance decreases
        proximity_factor = 1.0 - dist_next / near_target_threshold  # 0 at threshold, 1 at target
        angle_penalty = 0.1 * abs(body_angle) * proximity_factor
        angular_penalty = 0.05 * abs(angular_vel) * proximity_factor
        speed_penalty = 0.2 * speed * proximity_factor
        stability_penalty = -(angle_penalty + angular_penalty + speed_penalty)
    else:
        stability_penalty = 0.0
    
    # ========== Total Reward ==========
    total_reward = progress_reward + distance_reward + stability_penalty
    
    # ========== Components Dictionary ==========
    components = {
        "progress_reward": progress_reward,
        "distance_reward": distance_reward,
        "stability_penalty": stability_penalty,
        "total_reward": total_reward
    }
    
    return float(total_reward), components