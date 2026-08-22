def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Bipedal locomotion reward for rough terrain:
    - Primary: forward velocity reward with a compound health gate
      that depends on both hull angle and hull angular velocity.
    - Constraint: hinge tilt penalty (unchanged).
    """

    # ==================== Extract signals ====================
    next_hull_angle   = next_obs[0]
    next_hull_angvel  = next_obs[1]                     # newly used
    horizontal_speed  = next_obs[2]

    # ==================== Constants ====================
    # Tilt gate thresholds (unchanged)
    TILT_CRITICAL      = 0.6
    TILT_WARNING_START = 0.25
    TILT_WARNING_MARGIN = 0.35

    # New angvel gate thresholds
    SAFE_ANGVEL    = 0.5
    CRITICAL_ANGVEL = 1.0

    # Hinge penalty constants (unchanged)
    SAFE_TILT   = 0.3
    TILT_WEIGHT = 0.5

    # Forward weight (unchanged)
    FORWARD_WEIGHT = 2.0

    # ==================== Compound gate ====================
    # -- Tilt factor (identical to previous version) --
    abs_tilt = abs(next_hull_angle)
    if abs_tilt <= TILT_WARNING_START:
        tilt_gate = 1.0
    elif abs_tilt >= TILT_CRITICAL:
        tilt_gate = 0.0
    else:
        tilt_gate = (TILT_CRITICAL - abs_tilt) / TILT_WARNING_MARGIN

    # -- Angular velocity factor (new) --
    abs_angvel = abs(next_hull_angvel)
    if abs_angvel <= SAFE_ANGVEL:
        angvel_gate = 1.0
    elif abs_angvel >= CRITICAL_ANGVEL:
        angvel_gate = 0.0
    else:
        angvel_gate = (CRITICAL_ANGVEL - abs_angvel) / (CRITICAL_ANGVEL - SAFE_ANGVEL)

    gate = tilt_gate * angvel_gate

    # ==================== Component A: Gated forward progress ====================
    forward_reward  = FORWARD_WEIGHT * horizontal_speed ** 2
    gated_forward   = gate * forward_reward

    # ==================== Component B: Hinge tilt penalty (unchanged) ====================
    tilt_excess   = max(0.0, abs_tilt - SAFE_TILT)
    tilt_penalty  = TILT_WEIGHT * tilt_excess
    stability_hinge_penalty = -tilt_penalty

    # ==================== Total reward ====================
    total_reward = gated_forward + stability_hinge_penalty

    # ==================== Components dict ====================
    components = {
        'gated_forward_speed': gated_forward,
        'stability_tilt_hinge_penalty': stability_hinge_penalty
    }

    return float(total_reward), components