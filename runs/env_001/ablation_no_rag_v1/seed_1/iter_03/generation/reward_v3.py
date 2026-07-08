def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract positions
    x_curr, y_curr = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]
    body_angle = obs[4]

    # Euclidean distances to target platform
    dist_curr = (x_curr ** 2 + y_curr ** 2) ** 0.5
    dist_next = (x_next ** 2 + y_next ** 2) ** 0.5
    progress = dist_curr - dist_next

    # Stability: penalise large body tilt
    tilt_penalty = -0.1 * abs(body_angle)

    # Controlled descent: reward upward acceleration (deceleration against gravity)
    vy_curr = obs[3]
    vy_next = next_obs[3]
    delta_vy = vy_next - vy_curr
    # Only reward deceleration (positive delta_vy), avoid penalising natural downward acceleration
    vertical_accel_reward = 0.1 * max(0.0, delta_vy)

    total_reward = progress + tilt_penalty + vertical_accel_reward

    components = {
        'distance_progress': progress,
        'tilt_penalty': tilt_penalty,
        'vertical_accel_reward': vertical_accel_reward
    }

    return float(total_reward), components