def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- signal extraction ----
    body_z   = obs[0]
    quat_w   = obs[1]
    quat_x   = obs[2]
    quat_y   = obs[3]
    v_x      = obs[13]  # forward velocity
    v_y      = obs[14]  # lateral velocity

    # ---- upright projection (continuous, always gradient) ----
    up_z = 1.0 - 2.0 * (quat_x ** 2 + quat_y ** 2)  # 1.0 when perfectly upright, -1.0 when inverted

    # ---- forward progress (direct, NO gate) ----
    w_fwd   = 1.0
    forward = w_fwd * v_x

    # ---- body height safety (hinge quadratic penalty, softened) ----
    z_low_safe  = 0.35
    z_high_safe = 0.85
    w_h       = 1.0  # was 10.0
    low_hinge = max(0.0, z_low_safe - body_z)
    high_hinge= max(0.0, body_z - z_high_safe)
    height_penalty = -w_h * (low_hinge ** 2 + high_hinge ** 2)

    # ---- upright guidance (continuous gentle quadratic penalty) ----
    # Guides uprightness at every step without gate-killing exploration
    w_up          = 0.5  # was 5.0 + hinge
    upright_error = (1.0 - up_z)  # 0.0 when upright, 2.0 when inverted
    upright_penalty = -w_up * (upright_error ** 2)  # quadratic: gentle near upright, steep near fall

    # ---- lateral stability (quadratic penalty, unchanged) ----
    w_lat          = 0.2
    lateral_penalty = -w_lat * (v_y ** 2)

    # ---- action magnitude (light energy/smoothness proxy, unchanged) ----
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