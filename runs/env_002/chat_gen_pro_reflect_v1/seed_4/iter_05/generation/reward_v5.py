def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ============================================================
    # Signal extraction
    # ============================================================
    horizontal_vel = obs[2]
    hull_angle = obs[0]

    # ============================================================
    # Component 1: forward_progress (main learning signal)
    # ============================================================
    forward_reward = 1.0 * horizontal_vel

    # ============================================================
    # Component 2: soft_health_gate (safety: attenuates forward reward
    #   when balance degrades; gate = 1.0 for |angle| ≤ 0.25 rad)
    # ============================================================
    gate_lower = 0.25
    gate_upper = 0.5
    gate_raw = max(0.0, 1.0 - (abs(hull_angle) - gate_lower) / (gate_upper - gate_lower))
    gate_factor = gate_raw
    gated_forward = forward_reward * gate_factor

    # ============================================================
    # Component 3: energy_penalty (scaled to ~3% of main signal)
    # ============================================================
    action_power = action[0]**2 + action[1]**2 + action[2]**2 + action[3]**2
    energy_penalty = -0.02 * action_power

    # ============================================================
    # Component 4 (NEW): angle_hinge_penalty
    #   Adds posture correction when body lean approaches fall
    #   threshold (0.5 rad). Penalty = 0.5 * (|angle| - 0.4)^2
    #   for |angle| > 0.4, else 0. This protects safe erect postures
    #   while nudging risky leans back toward stability.
    # ============================================================
    angle_threshold = 0.4
    angle_excess = max(0.0, abs(hull_angle) - angle_threshold)
    angle_hinge_penalty = -0.5 * angle_excess ** 2

    # ============================================================
    # Total reward
    # ============================================================
    total_reward = gated_forward + energy_penalty + angle_hinge_penalty

    # ============================================================
    # Components dict
    # ============================================================
    components = {
        'forward_reward': forward_reward,
        'gate_factor': gate_factor,
        'gated_forward': gated_forward,
        'energy_penalty': energy_penalty,
        'angle_hinge_penalty': angle_hinge_penalty,
    }

    return float(total_reward), components