def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Observables
    x_pos = next_obs[0]
    y_pos = next_obs[1]
    x_vel = next_obs[2]
    y_vel = next_obs[3]
    body_angle = next_obs[4]
    angular_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 1. Distance reward: bounded exponential saturation to prevent rushing
    #    Replaces linear -distance which created a local optimum of fast crashing approach.
    #    -2*(1 - e^(-d/2)) asymptotes at -2.0, gradient at d=0 is -1.0 (same as original).
    distance_to_target = (x_pos ** 2 + y_pos ** 2) ** 0.5
    distance_reward = -2.0 * (1.0 - 2.718281828 ** (-distance_to_target / 2.0))

    # 2. Light stability penalty (preserved from best, iter 6)
    stability_penalty = -(
        0.15 * abs(x_vel) +
        0.05 * abs(y_vel) +
        0.2 * abs(body_angle) +
        0.2 * abs(angular_vel)
    )

    # 3. Landing quality: sum-based joint satisfaction (preserved from best, iter 6)
    prox_factor = max(0.0, 1.0 - distance_to_target / 2.0)
    speed_x_factor = max(0.0, 1.0 - abs(x_vel) / 0.8)
    speed_y_factor = max(0.0, 1.0 - abs(y_vel) / 0.8)
    angle_factor = max(0.0, 1.0 - abs(body_angle) / 0.4)
    contact_factor = 0.2 + 0.8 * (left_contact + right_contact) / 2.0

    sum_of_factors = prox_factor + speed_x_factor + speed_y_factor + angle_factor + contact_factor
    landing_quality = 0.2 * sum_of_factors

    total_reward = distance_reward + stability_penalty + landing_quality

    components = {
        "distance_reward": distance_reward,
        "stability_penalty": stability_penalty,
        "landing_quality": landing_quality
    }

    return float(total_reward), components