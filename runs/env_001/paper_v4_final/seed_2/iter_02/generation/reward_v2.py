def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack next_obs dimensions per environment card
    x_pos = next_obs[0]
    y_pos = next_obs[1]
    x_vel = next_obs[2]
    y_vel = next_obs[3]
    body_angle = next_obs[4]
    angular_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 1. Primary progress signal: dense quadratic penalty on position and velocity errors.
    #    Encourages the lander to move toward (0,0) with zero speed.
    pos_sq_error = x_pos**2 + y_pos**2
    vel_sq_error = x_vel**2 + y_vel**2
    progress = -0.05 * pos_sq_error - 0.1 * vel_sq_error

    # 2. Stability constraint: quadratic penalty on body angle and angular velocity.
    #    Keeps the lander upright and prevents excessive spinning.
    pose_penalty = -5.0 * (body_angle**2) - 0.5 * (angular_vel**2)

    # 3. Soft landing bonus: now uses a continuous contact factor instead of a hard gate.
    #    contact_factor = (left_contact + right_contact) / 2.0, so single-leg contact gives 0.5.
    contact_factor = (left_contact + right_contact) / 2.0
    speed_magnitude = abs(x_vel) + abs(y_vel)
    speed_factor = 1.0 / (1.0 + 5.0 * speed_magnitude)
    angle_factor = 1.0 / (1.0 + 20.0 * abs(body_angle))
    landing_bonus = 10.0 * contact_factor * speed_factor * angle_factor

    total_reward = progress + pose_penalty + landing_bonus

    components = {
        'progress': progress,
        'pose_penalty': pose_penalty,
        'landing_bonus': landing_bonus
    }
    return float(total_reward), components