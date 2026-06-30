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
    # Increased coefficient from 100 to 300 to dominate and overcome original_env_reward
    progress_reward = 300.0 * progress_delta
    
    # ========== Component 2: Stability Penalty (reduced by 50% to avoid dominance) ==========
    # Penalize high speed, large angle, and high angular velocity
    speed = (x_vel ** 2 + y_vel ** 2) ** 0.5
    # Reduced all penalties by 50% compared to previous
    angle_penalty = 0.025 * abs(body_angle)
    angular_penalty = 0.01 * abs(angular_vel)
    speed_penalty = 0.05 * speed
    
    stability_penalty = -(angle_penalty + angular_penalty + speed_penalty)
    
    # ========== Component 3: Soft Landing Shaping (increased coefficient and relaxed threshold) ==========
    # Continuous shaping that rewards being near target, low speed, stable angle, and both contacts
    # Relaxed near_target threshold from 0.5 to 1.0 to increase trigger rate
    near_target_score = max(0.0, 1.0 - dist_next / 1.0)  # 1.0 when dist=0, 0.0 when dist>=1.0
    low_speed_score = max(0.0, 1.0 - speed / 0.3)  # 1.0 when speed=0, 0.0 when speed>=0.3
    stable_angle_score = max(0.0, 1.0 - abs(body_angle) / 0.2)  # 1.0 when angle=0, 0.0 when angle>=0.2
    both_contact_score = 1.0 if (left_contact > 0.5) and (right_contact > 0.5) else 0.0
    
    # Increased coefficient from 2.0 to 10.0 to make it meaningful when triggered
    landing_shaping = 10.0 * near_target_score * low_speed_score * stable_angle_score * (0.5 + 0.5 * both_contact_score)
    
    # ========== Total Reward ==========
    total_reward = progress_reward + stability_penalty + landing_shaping
    
    # ========== Components Dictionary ==========
    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "landing_shaping": landing_shaping,
        "total_reward": total_reward
    }
    
    return float(total_reward), components