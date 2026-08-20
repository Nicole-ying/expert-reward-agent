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

    # Landing quality: reward based on vertical speed and body angle when both legs contact
    left_contact = next_obs[6]
    right_contact = next_obs[7]
    if left_contact and right_contact:
        # Use vertical velocity (next_obs[3]) and body angle (next_obs[4])
        speed_angle_sum = abs(next_obs[3]) + abs(next_obs[4])
        landing_quality = 2.0 / (1.0 + speed_angle_sum)
    else:
        landing_quality = 0.0

    total_reward = progress + tilt_penalty + landing_quality

    components = {
        'distance_progress': progress,
        'tilt_penalty': tilt_penalty,
        'landing_quality': landing_quality
    }

    return float(total_reward), components