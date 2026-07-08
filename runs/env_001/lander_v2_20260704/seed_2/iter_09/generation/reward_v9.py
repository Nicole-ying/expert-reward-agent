def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Current state observables
    x_pos_curr = obs[0]
    y_pos_curr = obs[1]
    x_vel_curr = obs[2]
    y_vel_curr = obs[3]
    body_angle_curr = obs[4]
    left_contact_curr = obs[6]
    right_contact_curr = obs[7]

    # Next state observables
    x_pos = next_obs[0]
    y_pos = next_obs[1]
    x_vel = next_obs[2]
    y_vel = next_obs[3]
    body_angle = next_obs[4]
    angular_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 1. Distance reward: bounded exponential saturation to prevent rushing
    distance_to_target = (x_pos ** 2 + y_pos ** 2) ** 0.5
    distance_reward = -2.0 * (1.0 - 2.718281828 ** (-distance_to_target / 2.0))

    # 2. Light stability penalty
    stability_penalty = -(
        0.15 * abs(x_vel) +
        0.05 * abs(y_vel) +
        0.2 * abs(body_angle) +
        0.2 * abs(angular_vel)
    )

    # 3. Landing quality: improvement-based (potential difference)
    #    Measures how much the agent improved its landing quality this step.
    #    Persistent hovering yields zero, eliminating the state-reward exploit.
    curr_dist = (x_pos_curr ** 2 + y_pos_curr ** 2) ** 0.5
    curr_prox = max(0.0, 1.0 - curr_dist / 2.0)
    curr_speed_x = max(0.0, 1.0 - abs(x_vel_curr) / 0.8)
    curr_speed_y = max(0.0, 1.0 - abs(y_vel_curr) / 0.8)
    curr_angle = max(0.0, 1.0 - abs(body_angle_curr) / 0.4)
    curr_contact = 0.2 + 0.8 * (left_contact_curr + right_contact_curr) / 2.0
    curr_sum = curr_prox + curr_speed_x + curr_speed_y + curr_angle + curr_contact

    next_prox = max(0.0, 1.0 - distance_to_target / 2.0)
    next_speed_x = max(0.0, 1.0 - abs(x_vel) / 0.8)
    next_speed_y = max(0.0, 1.0 - abs(y_vel) / 0.8)
    next_angle = max(0.0, 1.0 - abs(body_angle) / 0.4)
    next_contact = 0.2 + 0.8 * (left_contact + right_contact) / 2.0
    next_sum = next_prox + next_speed_x + next_speed_y + next_angle + next_contact

    landing_quality = 5.0 * (next_sum - curr_sum)

    total_reward = distance_reward + stability_penalty + landing_quality

    components = {
        "distance_reward": distance_reward,
        "stability_penalty": stability_penalty,
        "landing_quality": landing_quality
    }

    return float(total_reward), components