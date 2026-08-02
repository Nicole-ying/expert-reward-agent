def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack next observation
    next_x = next_obs[0]
    next_y = next_obs[1]
    next_angle = next_obs[4]
    next_vx = next_obs[2]
    next_vy = next_obs[3]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # --- Progress: distance reduction ---
    dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_x ** 2 + next_y ** 2) ** 0.5
    w_progress = 1.0
    progress = dist - next_dist

    # --- Landing incentive: only when legs touch ground ---
    leg_contact = 1.0 if (left_contact > 0.5 or right_contact > 0.5) else 0.0
    # Speed magnitude (linear velocities)
    speed = (next_vx ** 2 + next_vy ** 2) ** 0.5
    speed_factor = 1.0 / (1.0 + 3.0 * speed)
    w_landing = 1.0
    landing_incentive = leg_contact * w_landing / (1.0 + next_dist * 5.0) * speed_factor

    # --- Health constraint: body angle ---
    w_angle = 0.5
    safe_angle = 0.3
    angle_error = abs(next_angle) - safe_angle
    angle_penalty = -w_angle * angle_error if angle_error > 0 else 0.0

    # --- Total reward ---
    total_reward = w_progress * progress + landing_incentive + angle_penalty

    components = {
        "progress_reward": w_progress * progress,
        "landing_incentive": landing_incentive,
        "angle_penalty": angle_penalty
    }
    return float(total_reward), components