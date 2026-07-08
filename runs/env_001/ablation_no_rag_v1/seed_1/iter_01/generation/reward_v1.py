def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract current and next positions
    x_curr, y_curr = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]
    body_angle = obs[4]  # current body angle

    # Euclidean distances to the target platform
    dist_curr = (x_curr ** 2 + y_curr ** 2) ** 0.5
    dist_next = (x_next ** 2 + y_next ** 2) ** 0.5

    # Main learning signal: progress toward the target
    progress = dist_curr - dist_next

    # Stability constraint: penalize large body tilt
    tilt_penalty = -0.1 * abs(body_angle)

    total_reward = progress + tilt_penalty

    components = {
        'distance_progress': progress,
        'tilt_penalty': tilt_penalty
    }

    return float(total_reward), components