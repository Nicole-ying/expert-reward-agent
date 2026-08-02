def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v1 reward for 2D lander: approach target pad with safe attitude.
    Main progress: reduction in Euclidean distance to goal (0,0).
    Safety constraints: hinge penalty on large body angle and angular velocity.
    """
    # Unpack observations
    x, y = obs[0], obs[1]
    next_x, next_y = next_obs[0], next_obs[1]
    angle = obs[4]
    angular_vel = obs[5]

    # ---------- 1) Main progress: distance reduction ----------
    dist = (x ** 2 + y ** 2) ** 0.5
    next_dist = (next_x ** 2 + next_y ** 2) ** 0.5
    delta_dist = dist - next_dist           # positive when moving toward target
    progress = 1.0 * delta_dist             # main dense signal

    # ---------- 2) Safety constraints (hinge) ----------
    # Body angle – only penalise when beyond safe limit
    safe_angle = 0.5                        # radians, tuned for early training safety
    angle_error = max(0.0, abs(angle) - safe_angle)
    angle_penalty = -0.5 * angle_error

    # Angular velocity – discourage violent spinning
    safe_angvel = 1.0
    angvel_error = max(0.0, abs(angular_vel) - safe_angvel)
    angvel_penalty = -0.1 * angvel_error

    # ---------- Total reward ----------
    total_reward = progress + angle_penalty + angvel_penalty

    components = {
        'progress': progress,
        'angle_penalty': angle_penalty,
        'angvel_penalty': angvel_penalty
    }
    return float(total_reward), components