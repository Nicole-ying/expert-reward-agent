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
    
    # 2. Stability penalty - reduced by 20x and distance-gated
    # Only penalize instability when near the target (dist < 3.0)
    # Far from target, let the agent move freely
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    angle_penalty = -0.025 * abs(next_body_angle)
    angular_vel_penalty = -0.015 * abs(next_angular_vel)
    speed_penalty = -0.01 * speed
    
    # Distance gate: only apply stability penalty when near target
    gate = 1.0 / (1.0 + 2.0 * next_dist)  # ~1 when dist=0, ~0.2 when dist=2, ~0.09 when dist=5
    stability_penalty = gate * (angle_penalty + angular_vel_penalty + speed_penalty)
    
    # 3. Soft landing proxy: continuous product of bounded factors
    # Each factor is in [0,1], product gives smooth gradient
    proximity_factor = 1.0 / (1.0 + 5.0 * next_dist)  # bounded [0,1], high when near target
    speed_factor = 1.0 / (1.0 + 5.0 * speed)  # bounded [0,1], high when slow
    angle_factor = 1.0 / (1.0 + 10.0 * abs(next_body_angle))  # bounded [0,1], high when upright
    angular_vel_factor = 1.0 / (1.0 + 5.0 * abs(next_angular_vel))  # bounded [0,1], low angular vel
    
    # Contact factor: both feet on ground
    contact_factor = (next_left_contact > 0.5) and (next_right_contact > 0.5)
    contact_factor = float(contact_factor)  # 0 or 1, but that's okay as a gate
    
    soft_landing_bonus = 5.0 * proximity_factor * speed_factor * angle_factor * angular_vel_factor * contact_factor
    
    # 4. Small energy penalty for using engines (action != 0)
    energy_penalty = 0.0
    if action != 0:
        energy_penalty = -0.05
    
    # Combine rewards
    total_reward = progress_delta_reward + stability_penalty + soft_landing_bonus + energy_penalty
    
    components = {
        "progress_delta_reward": progress_delta_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_bonus": soft_landing_bonus,
        "energy_penalty": energy_penalty,
        "total_reward": total_reward
    }
    
    return float(total_reward), components