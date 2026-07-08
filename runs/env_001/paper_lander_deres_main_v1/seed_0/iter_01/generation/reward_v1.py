def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack current and next states
    x, y = obs[0], obs[1]
    next_x, next_y = next_obs[0], next_obs[1]
    next_x_vel = next_obs[2]
    next_y_vel = next_obs[3]
    next_angle = next_obs[4]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 1. Core learning signal: progress towards target (0,0)
    dist_current = (x ** 2 + y ** 2) ** 0.5
    dist_next = (next_x ** 2 + next_y ** 2) ** 0.5
    progress = dist_current - dist_next  # positive when approaching

    # 2. Soft proxy for landing quality (encourages gentle, two‑legged touchdown)
    contact_term = left_contact * right_contact                     # both feet must touch
    pos_x_score = max(0.0, 1.0 - abs(next_x) / 1.0)
    pos_y_score = max(0.0, 1.0 - abs(next_y) / 1.0)
    pos_score = pos_x_score * pos_y_score
    vel_x_score = max(0.0, 1.0 - abs(next_x_vel) / 1.0)
    vel_y_score = max(0.0, 1.0 - abs(next_y_vel) / 1.0)
    vel_score = vel_x_score * vel_y_score
    angle_score = max(0.0, 1.0 - abs(next_angle) / 1.5)
    landing_quality = contact_term * pos_score * vel_score * angle_score  # [0, 1]

    # 3. Orientation penalty (gentle, keeps the craft upright)
    attitude_penalty = -0.01 * abs(next_angle)

    # Combine
    w_landing = 0.2
    total_reward = progress + w_landing * landing_quality + attitude_penalty

    components = {
        "progress_reward": progress,
        "landing_quality_reward": landing_quality,
        "attitude_penalty": attitude_penalty,
    }

    return float(total_reward), components