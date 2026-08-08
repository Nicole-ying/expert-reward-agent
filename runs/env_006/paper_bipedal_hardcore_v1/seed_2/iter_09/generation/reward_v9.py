def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- extract useful signals ----------
    horizontal_speed = obs[2]
    hull_angle = obs[0]
    hull_angular_velocity = obs[1]

    # ---------- health gate: close to 1 when upright, decays when tilting ----------
    denom = 1.0 + 10.0 * hull_angle * hull_angle + 0.1 * hull_angular_velocity * hull_angular_velocity
    health_gate = 1.0 / denom

    # ---------- forward progress ----------
    fwd_speed = max(0.0, horizontal_speed)
    progress_component = 1.0 * fwd_speed * health_gate

    # ---------- action regularisation ----------
    action_sum_sq = action[0]*action[0] + action[1]*action[1] + action[2]*action[2] + action[3]*action[3]
    action_penalty = -0.01 * action_sum_sq

    # ---------- hinge balance penalty: explicit tilt cost beyond safe zone ----------
    tilt_magnitude = abs(hull_angle)
    safe_threshold = 0.4   # ~23 degrees
    excess_tilt = max(0.0, tilt_magnitude - safe_threshold)
    hinge_balance_penalty = -0.5 * excess_tilt

    # ---------- total reward ----------
    total_reward = progress_component + action_penalty + hinge_balance_penalty

    components = {
        "progress": progress_component,
        "action_penalty": action_penalty,
        "hinge_balance_penalty": hinge_balance_penalty
    }
    return float(total_reward), components