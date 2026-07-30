def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- signal extraction ----
    body_z   = obs[0]
    quat_w   = obs[1]
    quat_x   = obs[2]
    quat_y   = obs[3]
    # quat_z = obs[4]  # not used directly
    v_x      = obs[13]  # forward velocity
    v_y      = obs[14]  # lateral velocity

    # ---- upright projection ----
    up_z = 1.0 - 2.0 * (quat_x ** 2 + quat_y ** 2)

    # ---- soft health gate for forward reward ----
    z_low_safe  = 0.35
    z_high_safe = 0.85
    gate_z_low  = max(0.0, min(1.0, (body_z - 0.2) / (z_low_safe - 0.2)))
    gate_z_high = max(0.0, min(1.0, (1.0 - body_z) / (1.0 - z_high_safe)))
    gate_z = gate_z_low * gate_z_high

    up_min      = 0.5
    up_thr      = 0.7
    gate_up     = max(0.0, min(1.0, (up_z - up_min) / (up_thr - up_min)))
    health_gate = gate_z * gate_up

    # ---- forward progress (main learning signal) ----
    w_fwd   = 1.0
    forward = w_fwd * v_x * health_gate

    # ---- body height safety (hinge quadratic penalty) ----
    w_h       = 10.0
    low_hinge = max(0.0, z_low_safe - body_z)
    high_hinge= max(0.0, body_z - z_high_safe)
    height_penalty = -w_h * (low_hinge ** 2 + high_hinge ** 2)

    # ---- upright orientation (hinge quadratic penalty) ----
    w_up          = 5.0
    upright_error = max(0.0, up_thr - up_z)
    upright_penalty = -w_up * (upright_error ** 2)

    # ---- lateral stability (quadratic penalty) ----
    w_lat          = 0.2
    lateral_penalty = -w_lat * (v_y ** 2)

    # ---- action magnitude (light energy/smoothness proxy) ----
    w_act = 0.005
    action_penalty = -w_act * sum(a ** 2 for a in action) / len(action)

    # ---- total reward ----
    total_reward = (forward + height_penalty + upright_penalty +
                    lateral_penalty + action_penalty)

    components = {
        "forward":          forward,
        "height_penalty":   height_penalty,
        "upright_penalty":  upright_penalty,
        "lateral_penalty":  lateral_penalty,
        "action_penalty":   action_penalty
    }
    return float(total_reward), components