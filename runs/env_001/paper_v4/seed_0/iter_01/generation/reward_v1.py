def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observations
    x = obs[0]          # horizontal distance to target pad
    y = obs[1]          # vertical distance (height) above pad
    x_vel = obs[2]      # horizontal velocity
    y_vel = obs[3]      # vertical velocity (positive upward, downward negative)
    angle = obs[4]      # body angle (pitch)
    ang_vel = obs[5]    # angular velocity (not used in v1, reserved)
    left_contact = obs[6]
    right_contact = obs[7]

    # 1. Goal proximity: bounded, always active, encourages bringing x and y close to zero
    #    Using individual bounded forms to avoid dominating entire reward
    kx = 0.5
    ky = 1.0
    prox_x = 1.0 / (1.0 + kx * abs(x))
    prox_y = 1.0 / (1.0 + ky * abs(y))
    proximity = prox_x + prox_y

    # 2. Soft landing reward: joint‑condition proxy, active only when near the pad
    threshold_y = 0.2
    landing_weight = 2.0
    if y < threshold_y:
        # Contact factor: both legs must touch for full 1.0, continuous gradient
        contact_factor = (left_contact + right_contact) / 2.0
        # Speed factor: low total speed is desired
        speed_factor = 1.0 / (1.0 + 5.0 * (abs(x_vel) + abs(y_vel)))
        # Angle factor: body should be nearly upright
        angle_factor = 1.0 / (1.0 + 10.0 * abs(angle))
        landing_bonus = landing_weight * contact_factor * speed_factor * angle_factor
        landing_reward = landing_bonus
    else:
        landing_reward = 0.0

    # 3. Fuel efficiency penalty: simple linear penalty for any engine use
    fuel_penalty = -0.05 if action != 0 else 0.0

    total_reward = proximity + landing_reward + fuel_penalty

    components = {
        "proximity": proximity,
        "landing_reward": landing_reward,
        "fuel_penalty": fuel_penalty
    }

    return float(total_reward), components