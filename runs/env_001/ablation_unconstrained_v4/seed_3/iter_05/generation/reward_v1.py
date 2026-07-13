def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Hyperparameters
    w_progress = 1.5
    k_landing = 4.0
    w_contact = 0.3
    w_speed_landing = 0.2
    w_angle = 0.5
    w_global_speed = 0.01

    # Current and next distances to target pad (relative coordinates)
    dist_now = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5

    # --- Component A: progress toward target ---
    progress_reward = w_progress * (dist_now - dist_next)

    # --- Component B: soft‑landing (activated by proximity) ---
    landing_factor = 1.0 / (1.0 + k_landing * dist_next)
    contact_bonus = (next_obs[6] + next_obs[7]) * w_contact
    speed_penalty = w_speed_landing * (abs(next_obs[2]) + abs(next_obs[3]))
    angle_penalty = w_angle * abs(next_obs[4])
    soft_landing = landing_factor * (contact_bonus - speed_penalty - angle_penalty)

    # --- Component C: gentle global speed suppression ---
    global_speed_penalty = -w_global_speed * (abs(next_obs[2]) + abs(next_obs[3]))

    total_reward = progress_reward + soft_landing + global_speed_penalty
    components = {
        "progress_reward": progress_reward,
        "soft_landing": soft_landing,
        "global_speed_penalty": global_speed_penalty
    }
    return float(total_reward), components