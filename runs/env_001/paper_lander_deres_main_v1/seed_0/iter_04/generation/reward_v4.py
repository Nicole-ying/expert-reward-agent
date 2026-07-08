def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack current and next states
    x, y = obs[0], obs[1]
    next_x, next_y = next_obs[0], next_obs[1]
    next_x_vel = next_obs[2]
    next_y_vel = next_obs[3]
    next_angle = next_obs[4]

    # 1. Core learning signal: progress towards target (0,0)
    dist_current = (x ** 2 + y ** 2) ** 0.5
    dist_next = (next_x ** 2 + next_y ** 2) ** 0.5
    progress = dist_current - dist_next  # positive when approaching

    # 2. Continuous approach quality shaping
    #    Three bounded factors [0,1], product ensures all must be satisfied
    prox_factor = 1.0 / (1.0 + 5.0 * dist_next)
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    speed_factor = 1.0 / (1.0 + speed)
    angle_factor = 1.0 / (1.0 + 5.0 * abs(next_angle))
    approach_quality = prox_factor * speed_factor * angle_factor  # [0, 1], dense

    # 3. Orientation penalty (gentle, keeps the craft upright)
    attitude_penalty = -0.01 * abs(next_angle)

    # 4. Contact bonus: reward leg contact gated by proximity to target.
    #    prox_factor gate ensures contact only matters near the landing site;
    #    continuous gradient guides the agent to descend the last distance.
    left_contact = next_obs[6]
    right_contact = next_obs[7]
    contact_bonus = prox_factor * (left_contact + right_contact) * 0.05

    w_approach = 0.15  # unchanged from iter 3
    approach_quality_reward = w_approach * approach_quality
    total_reward = progress + approach_quality_reward + attitude_penalty + contact_bonus

    components = {
        "progress_reward": progress,
        "approach_quality_reward": approach_quality_reward,
        "attitude_penalty": attitude_penalty,
        "contact_bonus": contact_bonus,
    }

    return float(total_reward), components