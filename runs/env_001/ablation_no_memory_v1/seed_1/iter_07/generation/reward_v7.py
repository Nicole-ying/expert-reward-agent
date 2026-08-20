def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Previous observation
    px_prev = obs[0]
    py_prev = obs[1]
    prev_distance = (px_prev**2 + py_prev**2)**0.5

    # Next observation
    px = next_obs[0]
    py = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]

    # Distance to target pad center
    next_distance = (px**2 + py**2)**0.5
    # Speed magnitude
    speed = (vx**2 + vy**2)**0.5

    # 1. Main learning signal: progress delta reward
    #    Positive when approaching target, negative when retreating.
    progress_delta = 3.0 * (prev_distance - next_distance)

    # 2. Stability constraint: light penalty on high speed, large angle, high angular velocity
    #    Distance-gated: only active when agent is near the target pad (within ~2 units).
    raw_stability = -0.1 * speed - 0.05 * abs(angle) - 0.05 * abs(ang_vel)
    gate_radius = 2.0
    distance_gate = max(0.0, 1.0 - next_distance / gate_radius)
    stability_penalty = raw_stability * distance_gate

    # 3. Soft approaching proxy: geometric mean of nearness and slowness
    sigma_dist = 0.2
    sigma_speed = 0.3
    nearness = 2.718281828 ** (-(next_distance**2) / (2.0 * sigma_dist**2))
    slowness = 2.718281828 ** (-(speed**2) / (2.0 * sigma_speed**2))
    soft_landing_reward = 1.0 * (nearness * slowness)**0.5

    # 4. Fuel efficiency penalty: penalize any engine use to encourage minimal thrust
    #    Action 0 (no_engine) is free; actions 1/2/3 incur a small per-step cost.
    fuel_penalty = -0.02 if action != 0 else 0.0

    total_reward = progress_delta + stability_penalty + soft_landing_reward + fuel_penalty

    components = {
        'progress_delta': progress_delta,
        'stability_penalty': stability_penalty,
        'soft_landing_reward': soft_landing_reward,
        'fuel_penalty': fuel_penalty
    }

    return float(total_reward), components