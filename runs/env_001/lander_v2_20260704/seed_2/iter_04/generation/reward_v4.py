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

    # 1. Primary learning signal: continuous negative Euclidean distance to goal
    distance_to_target = (x_pos ** 2 + y_pos ** 2) ** 0.5
    distance_reward = -1.0 * distance_to_target

    # 2. Stability penalty: significantly reduced to avoid hindering necessary maneuvers
    #    Previous weights (0.15, 0.05, 0.2, 0.2) overly constrained early exploration.
    w_vx = 0.06
    w_vy = 0.02
    w_angle = 0.08
    w_angvel = 0.08
    stability_penalty = -(
        w_vx * abs(x_vel) +
        w_vy * abs(y_vel) +
        w_angle * abs(body_angle) +
        w_angvel * abs(angular_vel)
    )

    # 3. Gradual landing quality: geometric mean of bounded factors
    prox_factor = max(0.0, 1.0 - distance_to_target / 2.0)
    speed_x_factor = max(0.0, 1.0 - abs(x_vel) / 0.8)
    speed_y_factor = max(0.0, 1.0 - abs(y_vel) / 0.8)
    angle_factor = max(0.0, 1.0 - abs(body_angle) / 0.4)
    contact_factor = 0.2 + 0.8 * (left_contact + right_contact) / 2.0

    product_of_factors = prox_factor * speed_x_factor * speed_y_factor * angle_factor * contact_factor
    landing_quality = 0.8 * (product_of_factors ** 0.2)

    total_reward = distance_reward + stability_penalty + landing_quality

    components = {
        "distance_reward": distance_reward,
        "stability_penalty": stability_penalty,
        "landing_quality": landing_quality
    }

    return float(total_reward), components