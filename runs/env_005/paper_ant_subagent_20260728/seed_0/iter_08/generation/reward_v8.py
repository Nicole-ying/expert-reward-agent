def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- signal extraction ----
    body_z   = obs[0]
    quat_x   = obs[2]
    quat_y   = obs[3]
    v_x      = obs[13]
    v_y      = obs[14]

    # ---- upright measure (1.0 when perfectly upright) ----
    up_z = 1.0 - 2.0 * (quat_x ** 2 + quat_y ** 2)

    # ---- height gate: 1.0 in safe zone, decays to 0 near hard limits ----
    z_low  = 0.25   # termination boundary: z <= 0.2
    z_high = 0.95   # termination boundary: z >= 1.0
    z_safe_low  = 0.35
    z_safe_high = 0.85

    low_factor = (body_z - z_low) / (z_safe_low - z_low)
    low_factor = max(0.0, min(1.0, low_factor))

    high_factor = (z_high - body_z) / (z_high - z_safe_high)
    high_factor = max(0.0, min(1.0, high_factor))

    height_gate = low_factor * high_factor

    # ---- forward progress (main signal, gated only by height) ----
    w_fwd = 1.0
    forward_reward = w_fwd * v_x * height_gate

    # ---- upright penalty (hinge: only active when up_z < threshold) ----
    upright_threshold = 0.7
    upright_deficit = max(0.0, upright_threshold - up_z)
    w_up_pen = 0.5
    upright_penalty = -w_up_pen * upright_deficit

    # ---- lateral stability (quadratic penalty) ----
    w_lat = 0.3
    lateral_penalty = -w_lat * (v_y ** 2)

    # ---- action smoothness ----
    w_act = 0.005
    action_penalty = -w_act * sum(a ** 2 for a in action) / len(action)

    # ---- total reward ----
    total_reward = forward_reward + upright_penalty + lateral_penalty + action_penalty

    components = {
        "forward_reward":   forward_reward,
        "upright_penalty":  upright_penalty,
        "lateral_penalty":  lateral_penalty,
        "action_penalty":   action_penalty,
        "_height_gate":     height_gate
    }
    return float(total_reward), components