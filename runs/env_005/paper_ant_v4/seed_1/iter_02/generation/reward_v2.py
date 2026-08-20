def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- extract signals ----------
    body_z          = obs[0]
    qw, qx, qy, qz = obs[1], obs[2], obs[3], obs[4]
    body_x_vel      = obs[13]
    body_y_vel      = obs[14]
    roll_vel        = obs[16]
    pitch_vel       = obs[17]

    # compute torso uprightness (world z projection)
    body_up_z = 1.0 - 2.0 * (qx**2 + qy**2)

    # -------- 1. forward progress (bounded, scaled up) --------
    forward_reward = 20.0 * (body_x_vel / (1.0 + abs(body_x_vel)))

    # -------- 2. health height (soft boundary cost) --------
    height_penalty = 0.0
    if body_z < 0.3:
        height_penalty = -20.0 * (0.3 - body_z)
    elif body_z > 0.9:
        height_penalty = -20.0 * (body_z - 0.9)

    # -------- 3. upright stability (posture + angular velocity) --------
    upright_reward = 1.0 * body_up_z
    ang_penalty    = -0.005 * (roll_vel**2 + pitch_vel**2)
    upright_stability = upright_reward + ang_penalty

    # -------- 4. lateral drift penalty --------
    lateral_penalty = -0.1 * (body_y_vel**2)

    # -------- total reward and components --------
    total = forward_reward + height_penalty + upright_stability + lateral_penalty

    components = {
        "forward_progress":       forward_reward,
        "height_penalty":         height_penalty,
        "upright_stability":      upright_stability,
        "lateral_drift_penalty":  lateral_penalty
    }

    return float(total), components