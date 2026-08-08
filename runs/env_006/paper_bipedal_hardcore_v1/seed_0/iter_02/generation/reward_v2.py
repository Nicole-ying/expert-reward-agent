def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Bipedal locomotion reward for rough terrain:
    - Primary: forward velocity reward with soft health gate based on hull stability
    - Constraint: quadratic stability penalty on tilt and angular velocity
    """

    # ==================== Extract signals ====================
    # Next hull state (immediate future)
    next_hull_angle = next_obs[0]
    next_hull_angvel = abs(next_obs[1])   # angular velocity magnitude

    # Forward velocity (next step)
    horizontal_speed = next_obs[2]

    # ==================== Constants ====================
    # Gate thresholds (unchanged)
    TILT_CRITICAL = 0.6
    TILT_WARNING_START = 0.25
    TILT_WARNING_MARGIN = 0.35

    # Weights
    FORWARD_WEIGHT = 2.0
    STABILITY_ANGLE_WEIGHT = 2.0
    STABILITY_ANGVEL_WEIGHT = 1.0

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

    # ==================== Component B: Quadratic stability penalty ====================
    # Penalize any deviation from upright and any angular velocity
    # Quadratic form gives mild penalty near zero and rapid growth as tilt/velocity increase
    angle_penalty = STABILITY_ANGLE_WEIGHT * (next_hull_angle ** 2)
    angvel_penalty = STABILITY_ANGVEL_WEIGHT * (next_hull_angvel ** 2)
    stability_penalty = -(angle_penalty + angvel_penalty)

    # ==================== Total reward ====================
    total_reward = gated_forward + stability_penalty

    # ==================== Components dict ====================
    components = {
        'gated_forward_speed': gated_forward,
        'stability_quad_penalty': stability_penalty
    }

    return float(total_reward), components