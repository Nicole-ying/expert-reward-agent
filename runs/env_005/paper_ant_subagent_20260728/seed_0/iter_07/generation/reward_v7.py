def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- signal extraction ----
    body_z   = obs[0]
    quat_x   = obs[2]
    quat_y   = obs[3]
    v_x      = obs[13]
    v_y      = obs[14]

    # ---- upright measure (continuous, 1.0 when perfectly upright) ----
    up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)

    # ---- upright gate (soft gating of forward progress) ----
    gate_lower = 0.3
    gate = (up_z - gate_lower) / (1.0 - gate_lower)
    gate = max(0.0, min(1.0, gate))

    # ---- gated forward progress (main progress signal) ----
    w_fwd = 1.0
    gated_forward = w_fwd * v_x * gate

    # ---- upright bonus (direct posture reward, increased weight) ----
    w_up = 0.5
    upright_bonus = w_up * up_z

    # ---- body height safety (soft hinge quadratic) ----
    z_low_safe  = 0.3
    z_high_safe = 0.8
    w_h = 0.5
    low_hinge  = max(0.0, z_low_safe - body_z)
    high_hinge = max(0.0, body_z - z_high_safe)
    height_penalty = -w_h * (low_hinge**2 + high_hinge**2)

    # ---- lateral stability (mild quadratic, calibrated to ~0.1 per-step) ----
    w_lat = 0.3
    lateral_penalty = -w_lat * (v_y ** 2)

    # ---- action smoothing (light energy regularizer) ----
    w_act = 0.005
    action_penalty = -w_act * sum(a**2 for a in action) / len(action)

    # ---- total reward ----
    total_reward = (gated_forward + upright_bonus +
                    height_penalty + lateral_penalty +
                    action_penalty)

    components = {
        "gated_forward":      gated_forward,
        "upright_bonus":      upright_bonus,
        "height_penalty":     height_penalty,
        "lateral_penalty":    lateral_penalty,
        "action_penalty":     action_penalty
    }
    return float(total_reward), components