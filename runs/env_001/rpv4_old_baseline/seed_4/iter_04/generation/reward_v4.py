def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Current and next state
    x_curr, y_curr = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]
    vx_next, vy_next = next_obs[2], next_obs[3]
    angle_next = next_obs[4]
    angvel_next = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # Derived quantities
    dist_curr = (x_curr ** 2 + y_curr ** 2) ** 0.5
    dist_next = (x_next ** 2 + y_next ** 2) ** 0.5
    speed_next = (vx_next ** 2 + vy_next ** 2) ** 0.5

    # Weights and constants
    w_approach = 2.0          # shaping reward for closing distance to target
    w_speed = 0.5             # gentle penalty for high speed
    w_angle = 1.0             # penalty for body tilt
    w_angvel = 0.5            # penalty for angular velocity
    survival_penalty = -0.1   # small step penalty to avoid lingering
    main_engine_penalty = -0.3
    side_engine_penalty = -0.03

    # Contact detection
    both_legs_on_platform = 1.0 if (left_contact > 0.5 and right_contact > 0.5) else 0.0
    base_contact_bonus = 100.0 * both_legs_on_platform

    # Soft landing bonus (high reward for gentle, upright touchdown)
    soft_landing_condition = both_legs_on_platform and abs(vy_next) < 0.5 and abs(angle_next) < 0.1
    soft_contact_bonus = 300.0 * soft_landing_condition

    # Shaping: reward for moving closer to the target (potential-based)
    approach_reward = w_approach * (dist_curr - dist_next)

    # State penalties (guiding toward stability and low energy)
    speed_penalty = -w_speed * speed_next
    angle_penalty = -w_angle * (angle_next ** 2)
    angvel_penalty = -w_angvel * (angvel_next ** 2)

    # Engine (fuel) penalty
    engine_penalty = 0.0
    if action == 2:
        engine_penalty = main_engine_penalty
    elif action == 1 or action == 3:
        engine_penalty = side_engine_penalty

    # Total reward
    total_reward = (approach_reward +
                    speed_penalty +
                    angle_penalty +
                    angvel_penalty +
                    survival_penalty +
                    engine_penalty +
                    base_contact_bonus +
                    soft_contact_bonus)

    components = {
        "approach_reward": approach_reward,
        "speed_penalty": speed_penalty,
        "angle_penalty": angle_penalty,
        "angvel_penalty": angvel_penalty,
        "survival_penalty": survival_penalty,
        "engine_penalty": engine_penalty,
        "base_contact_bonus": base_contact_bonus,
        "soft_contact_bonus": soft_contact_bonus
    }
    return float(total_reward), components