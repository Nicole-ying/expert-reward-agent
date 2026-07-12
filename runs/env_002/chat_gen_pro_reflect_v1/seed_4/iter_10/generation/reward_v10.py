def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ============================================================
    # Signal extraction
    # ============================================================
    horizontal_vel = obs[2]
    hull_angle = obs[0]

    # ============================================================
    # Component 1: forward_progress (linear main signal)
    # ============================================================
    forward_reward = 1.0 * max(0.0, horizontal_vel)

    # ============================================================
    # Component 2: soft_health_gate (amplify when mostly upright, attenuate when tilted)
    #   gate ∈ [0.0, 1.4]:
    #     - |angle| ≤ 0.3: 1.4 → 1.0 (gentle amplification)
    #     - |angle| > 0.3: 1.0 → 0.0 at 0.5 rad (attenuation)
    # ============================================================
    abs_angle = abs(hull_angle)
    if abs_angle <= 0.3:
        gate_factor = 1.0 + 0.4 * (1.0 - abs_angle / 0.3)
    else:
        gate_factor = max(0.0, 1.0 - (abs_angle - 0.3) / 0.2)

    gated_forward = forward_reward * gate_factor

    # ============================================================
    # Component 3: energy_penalty (unchanged)
    # ============================================================
    action_power = action[0]**2 + action[1]**2 + action[2]**2 + action[3]**2
    energy_penalty = -0.02 * action_power

    # ============================================================
    # Total reward
    # ============================================================
    total_reward = gated_forward + energy_penalty

    components = {
        'forward_reward': forward_reward,
        'gate_factor': gate_factor,
        'gated_forward': gated_forward,
        'energy_penalty': energy_penalty,
    }

    return float(total_reward), components