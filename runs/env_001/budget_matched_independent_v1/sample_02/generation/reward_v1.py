def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v1 reward for 2D lander: approach target, decelerate near pad, keep level,
    settle softly on both legs.  Uses only observable signals; no info/terminal flags.
    """
    # ── state extraction ──
    x_old, y_old = obs[0], obs[1]
    x_new, y_new = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    angvel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ── 1. goal proximity: improvement delta ──
    dist_old = (x_old ** 2 + y_old ** 2) ** 0.5
    dist_new = (x_new ** 2 + y_new ** 2) ** 0.5
    distance_progress = dist_old - dist_new

    # ── 2. velocity damping: quadratic penalty with distance-gated weight ──
    # Weight ramps from ~0.01 (far away, encourage fast approach) to ~0.15 (near
    # target, enforce soft touchdown).  Uses rational scaling instead of linear
    # division to give a smoother transition zone.
    speed_sq = vx ** 2 + vy ** 2
    proximity_weight = 0.01 + 0.14 / (1.0 + 3.0 * dist_new ** 2)
    velocity_damping = -proximity_weight * speed_sq

    # ── 3. orientation penalty: quadratic on angle + angular velocity ──
    orientation_penalty = -0.08 * (angle ** 2) - 0.04 * (angvel ** 2)

    # ── 4. soft-landing bonus: joint-condition proxy with rational factors ──
    # All three factors use 1/(1 + k * x^2) — rational decay, gentler tails than
    # exponential, less prone to collapsing to zero far from the target.
    both_contact = left_contact * right_contact          # binary 0/1
    dist_factor = 1.0 / (1.0 + 5.0 * dist_new ** 2)
    vel_factor  = 1.0 / (1.0 + 10.0 * speed_sq)
    angle_factor = 1.0 / (1.0 + 15.0 * angle ** 2)

    soft_landing = 0.5 * both_contact * dist_factor * vel_factor * angle_factor

    # ── combine ──
    total = distance_progress + velocity_damping + orientation_penalty + soft_landing

    components = {
        "distance_progress": distance_progress,
        "velocity_damping": velocity_damping,
        "orientation_penalty": orientation_penalty,
        "soft_landing_bonus": soft_landing,
    }

    return float(total), components
