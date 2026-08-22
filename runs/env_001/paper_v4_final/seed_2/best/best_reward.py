def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack next_obs dimensions per environment card
    x_pos = next_obs[0]
    y_pos = next_obs[1]
    x_vel = next_obs[2]
    y_vel = next_obs[3]
    body_angle = next_obs[4]
    angular_vel = next_obs[5]
    # obs[6], obs[7] not used in this version

    # 1. Primary progress signal: dense quadratic penalty on position and velocity errors.
    pos_sq_error = x_pos**2 + y_pos**2
    vel_sq_error = x_vel**2 + y_vel**2
    progress = -0.05 * pos_sq_error - 0.1 * vel_sq_error

    # 2. Stability constraint: quadratic penalty on body angle and angular velocity.
    pose_penalty = -5.0 * (body_angle**2) - 0.5 * (angular_vel**2)

    # 3. Approach & soft landing bonus: now based on proximity to origin, not contact.
    #    proximity = 1 at (0,0), decays with squared distance.
    proximity = 1.0 / (1.0 + 10.0 * (x_pos**2 + y_pos**2))
    speed_magnitude = abs(x_vel) + abs(y_vel)
    speed_factor = 1.0 / (1.0 + 5.0 * speed_magnitude)
    angle_factor = 1.0 / (1.0 + 20.0 * abs(body_angle))
    landing_bonus = 2.0 * proximity * speed_factor * angle_factor

    total_reward = progress + pose_penalty + landing_bonus

    components = {
        'progress': progress,
        'pose_penalty': pose_penalty,
        'landing_bonus': landing_bonus
    }
    return float(total_reward), components