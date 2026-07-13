def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Helper: landing quality evaluated at a given observation
    def _landing_quality(o):
        height = o[1]
        near_ground = 1.0 / (1.0 + 5.0 * abs(height))
        speed = (o[2] ** 2 + o[3] ** 2) ** 0.5
        low_speed = 1.0 / (1.0 + speed)
        angle_abs = abs(o[4])
        upright = 1.0 / (1.0 + 3.0 * angle_abs)
        x_offset = abs(o[0])
        centered = 1.0 / (1.0 + 3.0 * x_offset)
        return 0.5 * near_ground * (2.0 * low_speed + 1.0 * upright + 1.0 * centered)

    # 1. Main learning signal: distance improvement × attitude gate (unchanged)
    dist_current = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next    = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_raw = dist_current - dist_next
    angle_abs_next = abs(next_obs[4])
    gate = 1.0 / (1.0 + 5.0 * angle_abs_next)
    shaped_progress = 5.0 * progress_raw * gate

    # 2. Safety constraint: angular velocity light penalty (unchanged)
    ang_vel_penalty = -0.1 * (next_obs[5] ** 2)

    # 3. Landing quality: state→improvement (state_to_improvement transform)
    #     Reward becoming better, not occupying a good state.
    quality_current = _landing_quality(obs)
    quality_next    = _landing_quality(next_obs)
    landing_quality_improvement = 50.0 * (quality_next - quality_current)

    total_reward = shaped_progress + ang_vel_penalty + landing_quality_improvement
    components = {
        "shaped_progress": shaped_progress,
        "angular_vel_penalty": ang_vel_penalty,
        "landing_quality_improvement": landing_quality_improvement
    }
    return (float(total_reward), components)