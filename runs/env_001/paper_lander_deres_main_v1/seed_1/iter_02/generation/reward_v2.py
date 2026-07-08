def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # === Progress reward (main learning signal) ===
    # Scale up by 30x: raw distance delta is ~0.004/step, too weak to guide flight.
    # At 30x, per-step progress ~0.12, providing meaningful gradient during descent
    # while still letting landing quality bonus (~0.37/step active) dominate terminal phase.
    x_prev, y_prev = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]

    dist_prev = (x_prev ** 2 + y_prev ** 2) ** 0.5
    dist_next = (x_next ** 2 + y_next ** 2) ** 0.5
    progress_reward = 30.0 * (dist_prev - dist_next)  # positive when getting closer

    # === Landing quality bonus (soft proxy for successful landing) ===
    left_contact = next_obs[6]
    right_contact = next_obs[7]
    both_contact = (left_contact > 0.5) and (right_contact > 0.5)

    landing_quality_bonus = 0.0
    if both_contact:
        # position error from center of the pad
        pos_err = dist_next
        # velocity magnitude
        vx, vy = next_obs[2], next_obs[3]
        vel = (vx ** 2 + vy ** 2) ** 0.5
        # absolute body angle
        angle = abs(next_obs[4])

        # exponential-based quality score (max ~1.0 for perfect landing)
        # temperatures control how quickly the score decays with imperfection
        temp_pos = 0.2
        temp_vel = 0.5
        temp_angle = 0.1
        quality = (2.718281828 ** (-pos_err / temp_pos)) * \
                  (2.718281828 ** (-vel / temp_vel)) * \
                  (2.718281828 ** (-angle / temp_angle))
        landing_quality_bonus = quality

    total_reward = progress_reward + landing_quality_bonus

    components = {
        "progress_reward": progress_reward,
        "landing_quality_bonus": landing_quality_bonus
    }
    return float(total_reward), components