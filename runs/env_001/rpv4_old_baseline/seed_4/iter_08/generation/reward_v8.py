def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract state
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

    # 1. Approach shaping: reward for getting closer to the target
    approach_reward = 10.0 * (dist_curr - dist_next)

    # 2. Soft landing reward (continuous, to give partial credit during approach)
    contact_ratio = (left_contact + right_contact) / 2.0
    # speed factor: 1 when speed=0, linearly decays to 0 at speed=0.3
    speed_factor = max(0.0, 1.0 - speed_next / 0.3)
    # angle factor: 1 when angle=0, linearly decays to 0 at abs(angle)=0.15
    angle_factor = max(0.0, 1.0 - abs(angle_next) / 0.15)
    landing_success_reward = 200.0 * contact_ratio * speed_factor * angle_factor

    # 3. Speed penalty: strongly discourages high speed (quadratic)
    speed_penalty = -0.1 * (speed_next ** 2)

    # 4. Attitude penalty: discourages large tilt
    angle_penalty = -0.2 * (angle_next ** 2)

    # 5. Angular velocity penalty: discourages spinning
    angvel_penalty = -0.1 * (angvel_next ** 2)

    # 6. Horizontal centering penalty: keep the craft above the platform
    centering_penalty = -0.05 * abs(x_next)

    # 7. Small per-step survival bonus
    survival_bonus = 0.05

    # 8. Engine/fuel penalties
    main_penalty = -0.5 if action == 2 else 0.0
    side_penalty = -0.1 if action in (1, 3) else 0.0
    engine_penalty = main_penalty + side_penalty

    total_reward = (approach_reward +
                    landing_success_reward +
                    speed_penalty +
                    angle_penalty +
                    angvel_penalty +
                    centering_penalty +
                    survival_bonus +
                    engine_penalty)

    components = {
        "approach_reward": approach_reward,
        "landing_success_reward": landing_success_reward,
        "speed_penalty": speed_penalty,
        "angle_penalty": angle_penalty,
        "angvel_penalty": angvel_penalty,
        "centering_penalty": centering_penalty,
        "survival_bonus": survival_bonus,
        "engine_penalty": engine_penalty
    }
    return float(total_reward), components