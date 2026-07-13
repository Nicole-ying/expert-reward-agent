def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Current state
    x, y = obs[0], obs[1]
    vx, vy = obs[2], obs[3]
    angle = obs[4]
    left, right = obs[6], obs[7]
    # Next state
    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    nangle = next_obs[4]
    nleft, nright = next_obs[6], next_obs[7]

    # 1. Progress: reduction in distance to target
    dist_curr = (x ** 2 + y ** 2) ** 0.5
    dist_next = (nx ** 2 + ny ** 2) ** 0.5
    progress = max(0.0, dist_curr - dist_next)

    # 2. Landing gate: contact factor × posture factor
    # contact_factor encourages at least one contact, more is better; floor 0.3
    contact_factor = 0.3 + 0.7 * (nleft + nright) / 2.0   # range [0.3, 1.0]
    # posture_factor selects for small angle and low speed
    angle_sq = nangle ** 2
    speed_sq = nvx ** 2 + nvy ** 2
    posture_factor = 2.71828 ** (-10.0 * angle_sq) * 2.71828 ** (-2.0 * speed_sq)
    landing_gate = contact_factor * posture_factor

    # 3. Main reward: progress modulated by landing quality
    main_reward = progress * landing_gate * 10.0

    # 4. Energy penalty (unchanged)
    energy_penalty = -0.01 if action != 0 else 0.0

    # 5. Terminal velocity penalty (unchanged from previous round)
    vel_penalty = 0.0
    if ny < 0.05 and abs(nx) < 0.1:
        if nvy < -0.3:
            vel_penalty = -0.5 * max(0.0, -nvy - 0.3)

    total_reward = main_reward + energy_penalty + vel_penalty

    components = {
        "progress_gate_reward": main_reward,
        "energy_penalty": energy_penalty,
        "terminal_velocity_penalty": vel_penalty
    }
    return float(total_reward), components