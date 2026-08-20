def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Positions
    x_curr, y_curr = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]
    body_angle = obs[4]

    # Distance progress toward target (0,0)
    dist_curr = (x_curr ** 2 + y_curr ** 2) ** 0.5
    dist_next = (x_next ** 2 + y_next ** 2) ** 0.5
    progress = dist_curr - dist_next

    # Stability: penalise large body tilt
    tilt_penalty = -0.1 * abs(body_angle)

    # Landing reward: positive every step both legs are on the platform
    left_contact = next_obs[6]
    right_contact = next_obs[7]
    landing_support = 2.0 if (left_contact and right_contact) else 0.0

    total_reward = progress + tilt_penalty + landing_support

    components = {
        'distance_progress': progress,
        'tilt_penalty': tilt_penalty,
        'landing_support': landing_support
    }

    return float(total_reward), components