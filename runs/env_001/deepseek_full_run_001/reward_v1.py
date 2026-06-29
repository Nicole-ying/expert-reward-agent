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
    
    # Goal is at (0, 0) in relative coordinates
    goal_x = 0.0
    goal_y = 0.0
    
    # ---- Component 1: Progress Delta Reward (main learning signal) ----
    # Reward for moving closer to the goal
    current_dist = ((x_pos - goal_x) ** 2 + (y_pos - goal_y) ** 2) ** 0.5
    next_dist = ((next_x_pos - goal_x) ** 2 + (next_y_pos - goal_y) ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_scale = 2.0
    progress_reward = progress_delta * progress_scale
    
    # ---- Component 2: Stability Penalty (light constraint) ----
    # Penalize high velocity, large body angle, and high angular velocity
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    angle_penalty_scale = 0.1
    speed_penalty_scale = 0.05
    angular_vel_penalty_scale = 0.05
    
    angle_penalty = -angle_penalty_scale * abs(next_body_angle)
    speed_penalty = -speed_penalty_scale * speed
    angular_vel_penalty = -angular_vel_penalty_scale * abs(next_angular_vel)
    
    stability_penalty = angle_penalty + speed_penalty + angular_vel_penalty
    
    # ---- Component 3: Soft Landing Proxy (small bonus for task completion proxy) ----
    # Bonus when near target, low speed, stable angle, and both supports contact
    near_target_threshold = 0.3
    low_speed_threshold = 0.3
    stable_angle_threshold = 0.2
    
    near_target = next_dist < near_target_threshold
    low_speed = speed < low_speed_threshold
    stable_angle = abs(next_body_angle) < stable_angle_threshold
    both_contact = (next_left_contact > 0.5) and (next_right_contact > 0.5)
    
    soft_landing_bonus = 0.0
    if near_target and low_speed and stable_angle and both_contact:
        soft_landing_bonus = 1.0
    
    # ---- Combine components ----
    total_reward = progress_reward + stability_penalty + soft_landing_bonus
    
    # ---- Build components dict ----
    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_bonus": soft_landing_bonus,
        "total_reward": total_reward
    }
    
    return float(total_reward), components