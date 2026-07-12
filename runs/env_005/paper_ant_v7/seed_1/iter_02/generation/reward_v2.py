def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation signals
    body_z = obs[0]                     # body height
    vx = obs[13]                        # forward velocity (world x)
    vy = obs[14]                        # lateral velocity (world y) - NEW
    quat_x = obs[2]
    quat_y = obs[3]

    # ---- 1. Survival gate: height factor ----
    low_safe  = max(0.0, min(1.0, (body_z - 0.2) / 0.2))
    high_safe = max(0.0, min(1.0, (1.0 - body_z) / 0.2))
    height_factor = min(low_safe, high_safe)

    # ---- 2. Survival gate: uprightness factor ----
    body_up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)
    upright_factor = max(0.0, min(1.0, body_up_z))

    survival_gate = height_factor * upright_factor

    # ---- 3. Main progress component (gated) ----
    w_progress = 1.0
    forward_gated = w_progress * vx * survival_gate

    # ---- 4. Lateral drift penalty (NEW) ----
    w_lateral = 0.2
    lateral_penalty = -w_lateral * (vy ** 2)

    # ---- 5. Energy efficiency term (coefficient corrected) ----
    w_energy = 0.01  # was 0.0005, now 20x stronger
    action_penalty = -w_energy * sum(a**2 for a in action)

    # Total reward
    total_reward = forward_gated + lateral_penalty + action_penalty
    components = {
        'forward_gated': forward_gated,
        'lateral_penalty': lateral_penalty,
        'action_penalty': action_penalty
    }
    return float(total_reward), components