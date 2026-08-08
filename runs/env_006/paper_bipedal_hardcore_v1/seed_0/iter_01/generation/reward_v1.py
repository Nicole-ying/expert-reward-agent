def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Bipedal locomotion reward for rough terrain:
    - Primary: forward velocity reward with soft health gate based on hull stability
    - Constraint: postural hinge penalty for extreme tilt

    role-based component budget: 2 components (forward_progress + postural_stability)
    """
    # ==================== Extract signals ====================
    # Current hull state
    hull_angle = obs[0]
    hull_angvel = abs(obs[1])

    # Next hull state
    next_hull_angle = next_obs[0]
    next_hull_angvel = abs(next_obs[1])

    # Forward velocity (next step)
    horizontal_speed = next_obs[2]

    # ==================== Constants ====================
    # Hull tilt safety thresholds (radians, empirically near falling boundary)
    TILT_CRITICAL = 0.6      # near-falling severe tilt
    TILT_WARNING_START = 0.25  # begin gentle attenuation
    TILT_WARNING_MARGIN = 0.35  # attenuation window width

    # Hinge penalty thresholds
    HINGE_THRESHOLD = 0.35   # start penalizing tilt above this
    HINGE_SCALE = 1.0

    # Weights (balanced for per-step magnitude comparable)
    FORWARD_WEIGHT = 2.0
    POSTURE_HINGE_WEIGHT = 0.5

    # ==================== Component A: Forward progress with soft health gate ====================
    # Gate factor: linear attenuation from 1.0 (safe) to 0.0 (critical)
    # Uses next_hull_angle (immediate future stability) to gate reward
    abs_tilt = abs(next_hull_angle)
    if abs_tilt <= TILT_WARNING_START:
        gate = 1.0
    elif abs_tilt >= TILT_CRITICAL:
        gate = 0.0
    else:
        gate = (TILT_CRITICAL - abs_tilt) / TILT_WARNING_MARGIN

    # Forward reward: convex to encourage speed, not just minimal forward motion
    forward_reward = FORWARD_WEIGHT * horizontal_speed ** 2
    gated_forward = gate * forward_reward

    # ==================== Component B: Postural hinge penalty ====================
    # Only penalize when tilt exceeds safe threshold
    # Penalizes both current and next tilt, and angular velocity
    current_excess = max(0.0, abs(hull_angle) - HINGE_THRESHOLD)
    next_excess = max(0.0, abs_tilt - HINGE_THRESHOLD)

    # Average excess over the step (current + next) with velocity penalty
    tilt_penalty = HINGE_SCALE * (current_excess + next_excess) * 0.5
    angvel_penalty = 0.3 * next_hull_angvel  # angular velocity contributes to instability

    posture_penalty = -POSTURE_HINGE_WEIGHT * (tilt_penalty + angvel_penalty)

    # ==================== Total reward ====================
    total_reward = gated_forward + posture_penalty

    # ==================== Components dict ====================
    components = {
        'gated_forward_speed': gated_forward,
        'posture_hinge_penalty': posture_penalty
    }

    return float(total_reward), components