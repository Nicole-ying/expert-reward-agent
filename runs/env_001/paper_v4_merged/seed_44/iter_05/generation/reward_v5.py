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

    # Soft landing guide (unchanged)
    dist_factor = 2.718281828 ** (-dist_next / 0.5)
    x_vel = next_obs[2]
    y_vel = next_obs[3]
    speed_factor = max(0.0, 1.0 - (abs(x_vel) + abs(y_vel)) / 1.0)
    landing_reward = dist_factor * speed_factor

    # Gated contact reward: reward simultaneous leg contact when close and slow
    left_contact = next_obs[6]
    right_contact = next_obs[7]
    both_contact = left_contact * right_contact
    proximity_gate = 2.718281828 ** (-dist_next / 0.2)
    speed_gate = max(0.0, 1.0 - (abs(x_vel) + abs(y_vel)) / 0.5)
    contact_reward = 0.1 * both_contact * proximity_gate * speed_gate

    total = (
        10.0 * progress
        - 0.5 * angle_penalty
        + 0.01 * landing_reward
        + contact_reward
    )

    components = {
        "progress": 10.0 * progress,
        "angle_penalty": -0.5 * angle_penalty,
        "landing_reward": 0.01 * landing_reward,
        "contact_reward": contact_reward
    }

    return float(total), components