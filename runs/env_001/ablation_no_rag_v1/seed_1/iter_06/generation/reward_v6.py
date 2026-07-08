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

    # Landing guidance: sparse bonus on both contacts + dense slow-down when not in contact
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    if left_contact and right_contact:
        landing_guidance = 2.0
    else:
        # Encourage slowing down when close to the ground (y small).
        # Penalty proportional to vertical speed, softened by height.
        vertical_speed = abs(next_obs[3])
        height = next_obs[1]  # relative height to the platform
        landing_guidance = -0.05 * vertical_speed / (1.0 + height)

    total_reward = progress + tilt_penalty + landing_guidance

    components = {
        'distance_progress': progress,
        'tilt_penalty': tilt_penalty,
        'landing_guidance': landing_guidance
    }

    return float(total_reward), components