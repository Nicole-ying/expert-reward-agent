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
    
    # ========== Component 1: Progress Delta Reward (main learning signal, significantly increased) ==========
    # Distance to target at current step
    dist_current = (x_pos ** 2 + y_pos ** 2) ** 0.5
    # Distance to target at next step
    dist_next = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    # Progress: positive when moving closer to target
    progress_delta = dist_current - dist_next
    # Significantly increased coefficient to overcome penalty dominance and drive learning
    progress_reward = 500.0 * progress_delta
    
    # ========== Component 2: Distance Reward (light anchor, provides baseline gradient) ==========
    # Continuous distance-based reward to encourage being close to target
    # Small weight to avoid dominating progress signal
    distance_reward = -0.1 * dist_next
    
    # ========== Component 3: Conditional Stability Penalty (only when near target) ==========
    # Penalize high speed, large angle, and high angular velocity
    speed = (x_vel ** 2 + y_vel ** 2) ** 0.5
    # Only apply stability penalty when close to target (dist_next < 2.0)
    # This avoids suppressing exploration when far away
    near_target_condition = 1.0 if dist_next < 2.0 else 0.0
    
    angle_penalty = 0.05 * abs(body_angle) * near_target_condition
    angular_penalty = 0.02 * abs(angular_vel) * near_target_condition
    speed_penalty = 0.1 * speed * near_target_condition
    
    stability_penalty = -(angle_penalty + angular_penalty + speed_penalty)
    
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