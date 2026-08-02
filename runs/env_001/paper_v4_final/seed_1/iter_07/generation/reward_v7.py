def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack next observation
    next_x = next_obs[0]
    next_y = next_obs[1]
    next_vx = next_obs[2]
    next_vy = next_obs[3]
    next_angle = next_obs[4]
    next_angvel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # --- Progress: distance reduction (coefficient increased) ---
    dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_x ** 2 + next_y ** 2) ** 0.5
    w_progress = 5.0
    progress = dist - next_dist

    # --- Landing incentive: only when legs touch ground ---
    leg_contact = 1.0 if (left_contact > 0.5 or right_contact > 0.5) else 0.0
    speed = (next_vx ** 2 + next_vy ** 2) ** 0.5
    speed_factor = 1.0 / (1.0 + 3.0 * speed)
    w_landing = 1.0
    landing_incentive = leg_contact * w_landing / (1.0 + next_dist * 5.0) * speed_factor

    # --- Angular velocity penalty (unchanged) ---
    w_angvel = 0.05
    safe_angvel = 0.5
    angvel_error = abs(next_angvel) - safe_angvel
    angvel_penalty = -w_angvel * angvel_error if angvel_error > 0 else 0.0

    # --- Total reward ---
    total_reward = w_progress * progress + landing_incentive + angvel_penalty

    components = {
        "progress_reward": w_progress * progress,
        "landing_incentive": landing_incentive,
        "angvel_penalty": angvel_penalty
    }
    return float(total_reward), components