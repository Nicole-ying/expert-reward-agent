def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward for 2D vehicle-like trajectory optimization:
    Reach and stabilize on the central target pad with minimal engine use.
    """
    # Previous observation
    px_prev = obs[0]
    py_prev = obs[1]
    prev_distance = (px_prev**2 + py_prev**2)**0.5
    prev_speed = (obs[2]**2 + obs[3]**2)**0.5

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

    # 3. Deceleration bonus: reward reducing speed when near the target.
    #    Replaces the previous soft_landing_reward product (nearness * slowness).
    #    Rewards the controlled action of braking, not the coincidental state.
    deceleration = max(0.0, prev_speed - speed)
    decel_gate_radius = 4.0
    decel_gate = max(0.0, 1.0 - next_distance / decel_gate_radius)
    deceleration_bonus = 2.0 * deceleration * decel_gate

    # Combine components
    total_reward = progress_delta + stability_penalty + deceleration_bonus

    components = {
        'progress_delta': progress_delta,
        'stability_penalty': stability_penalty,
        'deceleration_bonus': deceleration_bonus
    }

    return float(total_reward), components