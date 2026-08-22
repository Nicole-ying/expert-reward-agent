def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Bipedal locomotion reward for rough terrain:
    - Primary: forward velocity reward with soft health gate based on hull stability
    - Constraint: hinge-style tilt penalty that activates only beyond a safe angle.
    """

    # ==================== Extract signals ====================
    next_hull_angle = next_obs[0]
    horizontal_speed = next_obs[2]

    # ==================== Constants ====================
    # Gate thresholds (unchanged)
    TILT_CRITICAL = 0.6
    TILT_WARNING_START = 0.25
    TILT_WARNING_MARGIN = 0.35

    # Hinge penalty for tilt
    SAFE_TILT = 0.3          # below this no penalty
    TILT_WEIGHT = 0.5        # linear slope above SAFE_TILT

    # Forward weight
    FORWARD_WEIGHT = 2.0

    # ==================== Component A: Forward progress with soft health gate ====================
    abs_tilt = abs(next_hull_angle)
    if abs_tilt <= TILT_WARNING_START:
        gate = 1.0
    elif abs_tilt >= TILT_CRITICAL:
        gate = 0.0
    else:
        gate = (TILT_CRITICAL - abs_tilt) / TILT_WARNING_MARGIN

    forward_reward = FORWARD_WEIGHT * horizontal_speed ** 2
    gated_forward = gate * forward_reward

    # ==================== Component B: Hinge tilt stability penalty ====================
    # Penalize only when tilt exceeds SAFE_TILT, linearly up to CRITICAL.
    tilt_excess = max(0.0, abs_tilt - SAFE_TILT)
    tilt_penalty = TILT_WEIGHT * tilt_excess
    stability_hinge_penalty = -tilt_penalty

    # ==================== Total reward ====================
    total_reward = gated_forward + stability_hinge_penalty

    # ==================== Components dict ====================
    components = {
        'gated_forward_speed': gated_forward,
        'stability_tilt_hinge_penalty': stability_hinge_penalty
    }

    return float(total_reward), components