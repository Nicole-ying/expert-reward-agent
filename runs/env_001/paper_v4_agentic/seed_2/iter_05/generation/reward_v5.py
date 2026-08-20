def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v5 — replace dead landing_bonus with soft_approach_bonus using y distance gate
    and continuous velocity/angle factors. No contact dependency.
    """
    # ── current state ──
    x_cur = obs[0]
    y_cur = obs[1]
    vx_cur = obs[2]
    vy_cur = obs[3]
    angle_cur = obs[4]
    angvel_cur = obs[5]

    # ── next state ──
    x_next = next_obs[0]
    y_next = next_obs[1]

    # ── distance to pad (current) ──
    dist_cur  = (x_cur  ** 2 + y_cur  ** 2) ** 0.5
    dist_next = (x_next ** 2 + y_next ** 2) ** 0.5

    # ── weights / thresholds ──
    w_prox = 50.0
    w_vel  = 0.15
    w_ang  = 5.0
    proximity_threshold = 1.0

    w_approach = 10.0               # moderate bonus for near-ground stability

    # ── 1. Proximity delta ──
    proximity_delta = w_prox * (dist_cur - dist_next)

    # ── 2. Velocity danger ──
    speed_sq = vx_cur ** 2 + vy_cur ** 2
    velocity_danger = -w_vel * speed_sq / (dist_cur + proximity_threshold)

    # ── 3. Orientation penalty ──
    orientation_penalty = -w_ang * (angle_cur ** 2 + angvel_cur ** 2)

    # ── 4. Soft approach bonus (replaces landing_bonus) ──
    # Gate: how close we are to the pad vertically (y near 0)
    closeness = max(0.0, 1.0 - abs(y_next) / 0.5)   # active when |y| < 0.5

    # Speed factor: total speed should be low
    total_speed = (vx_cur ** 2 + vy_cur ** 2) ** 0.5
    speed_factor = max(0.0, 1.0 - total_speed / 1.0)   # active when speed < 1.0

    # Angle factor: body should be upright
    angle_factor = max(0.0, 1.0 - abs(angle_cur) / 0.5)  # active when |angle| < 0.5

    soft_approach_bonus = w_approach * closeness * speed_factor * angle_factor

    # ── Total reward ──
    total_reward = proximity_delta + velocity_danger + orientation_penalty + soft_approach_bonus

    components = {
        "proximity_delta": proximity_delta,
        "velocity_danger": velocity_danger,
        "orientation_penalty": orientation_penalty,
        "soft_approach_bonus": soft_approach_bonus,
    }
    return float(total_reward), components