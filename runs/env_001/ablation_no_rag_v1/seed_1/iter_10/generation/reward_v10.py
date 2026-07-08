def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x_curr, y_curr = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]
    body_angle = obs[4]

    dist_curr = (x_curr ** 2 + y_curr ** 2) ** 0.5
    dist_next = (x_next ** 2 + y_next ** 2) ** 0.5
    delta_dist = dist_curr - dist_next
    progress = 2.0 * delta_dist / (1.0 + abs(delta_dist))

    tilt_penalty = -0.1 * abs(body_angle)

    # Contact transitions
    prev_left = obs[6]
    prev_right = obs[7]
    curr_left = next_obs[6]
    curr_right = next_obs[7]

    prev_both_contact = (prev_left > 0.5) and (prev_right > 0.5)
    curr_both_contact = (curr_left > 0.5) and (curr_right > 0.5)

    # Touchdown reward when both legs first come into contact
    touchdown_bonus = 200.0 if (not prev_both_contact and curr_both_contact) else 0.0
    # Liftoff penalty when both legs are lost after contact (to prevent bonus exploit by bouncing)
    liftoff_penalty = -200.0 if (prev_both_contact and not curr_both_contact) else 0.0

    # Descent guidance: penalize vertical speed when legs are not in contact
    if curr_both_contact:
        descent_guidance = 0.0
    else:
        vertical_speed = abs(next_obs[3])
        height = next_obs[1]
        descent_guidance = -0.05 * vertical_speed / (1.0 + height)

    landing_guidance = touchdown_bonus + liftoff_penalty + descent_guidance

    total_reward = progress + tilt_penalty + landing_guidance

    components = {
        'distance_progress': progress,
        'tilt_penalty': tilt_penalty,
        'landing_guidance': landing_guidance
    }

    return float(total_reward), components