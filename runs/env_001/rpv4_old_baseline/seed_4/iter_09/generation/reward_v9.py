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

    # 1. Mild approach shaping (reduced coefficient to avoid overspeeding)
    approach_reward = 2.0 * (dist_curr - dist_next)

    # 2. Speed matching: encourage approach speed proportional to remaining distance
    desired_approach_speed = 0.15 * dist_next   # desired reduction per step
    actual_approach_reduction = dist_curr - dist_next
    speed_match_penalty = -3.0 * ((actual_approach_reduction - desired_approach_speed) ** 2)

    # 3. Soft landing reward: only when BOTH legs are in contact, speed and angle are small
    both_contact = left_contact * right_contact  # 1 only if both legs touch
    # speed factor: linearly decays from 1 at speed=0 to 0 at speed=0.2
    speed_factor = max(0.0, 1.0 - speed_next / 0.2)
    # angle factor: linearly decays from 1 at angle=0 to 0 at abs(angle)=0.1
    angle_factor = max(0.0, 1.0 - abs(angle_next) / 0.1)
    landing_success_reward = 400.0 * both_contact * speed_factor * angle_factor

    # 4. Speed penalty: strong quadratic discouragement of high speed
    speed_penalty = -1.5 * (speed_next ** 2)

    # 5. Attitude penalty
    angle_penalty = -0.5 * (angle_next ** 2)

    # 6. Angular velocity penalty
    angvel_penalty = -0.3 * (angvel_next ** 2)

    # 7. Horizontal centering: keep craft above platform center
    centering_penalty = -0.05 * abs(x_next)

    # 8. Small per-step survival bonus (encourages longer stable flight)
    survival_bonus = 0.05

    # 9. Engine/fuel penalties (reduced magnitudes)
    main_penalty = -0.2 if action == 2 else 0.0
    side_penalty = -0.05 if action in (1, 3) else 0.0
    engine_penalty = main_penalty + side_penalty

    total_reward = (approach_reward +
                    speed_match_penalty +
                    landing_success_reward +
                    speed_penalty +
                    angle_penalty +
                    angvel_penalty +
                    centering_penalty +
                    survival_bonus +
                    engine_penalty)

    components = {
        "approach_reward": approach_reward,
        "speed_match_penalty": speed_match_penalty,
        "landing_success_reward": landing_success_reward,
        "speed_penalty": speed_penalty,
        "angle_penalty": angle_penalty,
        "angvel_penalty": angvel_penalty,
        "centering_penalty": centering_penalty,
        "survival_bonus": survival_bonus,
        "engine_penalty": engine_penalty
    }
    return float(total_reward), components