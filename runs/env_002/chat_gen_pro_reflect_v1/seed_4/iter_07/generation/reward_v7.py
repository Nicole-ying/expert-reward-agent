def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ============================================================
    # Signal extraction
    # ============================================================
    horizontal_vel = obs[2]
    hull_angle = obs[0]

    # ============================================================
    # Component 1: forward_progress (linear main signal)
    #   Simple proportional reward for forward speed, bounded by
    #   the environment's natural velocity limits.
    # ============================================================
    forward_reward = 1.0 * max(0.0, horizontal_vel)

    # ============================================================
    # Component 2: soft_health_gate
    #   Attenuates forward reward when tilt approaches fall threshold.
    #   gate = 1.0 for |angle| ≤ 0.25, decays to 0.0 at |angle| ≥ 0.5.
    # ============================================================
    gate_lower = 0.25
    gate_upper = 0.5
    gate_raw = max(0.0, 1.0 - (abs(hull_angle) - gate_lower) / (gate_upper - gate_lower))
    gate_factor = gate_raw
    gated_forward = forward_reward * gate_factor

    # ============================================================
    # Component 3: energy_penalty (light action smoothness)
    # ============================================================
    action_power = action[0]**2 + action[1]**2 + action[2]**2 + action[3]**2
    energy_penalty = -0.02 * action_power

    # ============================================================
    # Total reward (angle_hinge_penalty removed – never triggered)
    # ============================================================
    total_reward = gated_forward + energy_penalty

    # ============================================================
    # Components dict
    # ============================================================
    components = {
        'forward_reward': forward_reward,
        'gate_factor': gate_factor,
        'gated_forward': gated_forward,
        'energy_penalty': energy_penalty,
    }

    return float(total_reward), components