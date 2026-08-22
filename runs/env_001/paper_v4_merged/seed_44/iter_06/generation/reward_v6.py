def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Progress toward center
    x_curr, y_curr = obs[0], obs[1]
    dist_curr = (x_curr ** 2 + y_curr ** 2) ** 0.5
    x_next, y_next = next_obs[0], next_obs[1]
    dist_next = (x_next ** 2 + y_next ** 2) ** 0.5
    progress = dist_curr - dist_next

    # Attitude and angular velocity penalty
    body_angle = next_obs[4]
    angular_vel = next_obs[5]
    angle_penalty = body_angle ** 2 + angular_vel ** 2

    # Soft landing guide
    dist_factor = 2.718281828 ** (-dist_next / 0.5)
    x_vel = next_obs[2]
    y_vel = next_obs[3]
    speed_factor = max(0.0, 1.0 - (abs(x_vel) + abs(y_vel)) / 1.0)
    landing_reward = dist_factor * speed_factor

    # One-time landing bonus: reward the moment both legs transition from non-contact to contact
    prev_contact = obs[6] * obs[7]          # 0 or 1
    next_contact = next_obs[6] * next_obs[7]
    contact_rising = max(0.0, next_contact - prev_contact)  # 1 on rising edge only

    proximity_factor = max(0.0, 1.0 - dist_next / 0.5)
    attitude_quality = max(0.0, 1.0 - (abs(body_angle) + abs(angular_vel)) / 0.5)
    speed_quality = max(0.0, 1.0 - (abs(x_vel) + abs(y_vel)) / 0.5)
    contact_landing_bonus = 20.0 * contact_rising * proximity_factor * attitude_quality * speed_quality

    total = (
        10.0 * progress
        - 0.5 * angle_penalty
        + 0.01 * landing_reward
        + contact_landing_bonus
    )

    components = {
        "progress": 10.0 * progress,
        "angle_penalty": -0.5 * angle_penalty,
        "landing_reward": 0.01 * landing_reward,
        "contact_landing_bonus": contact_landing_bonus
    }

    return float(total), components