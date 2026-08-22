def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v4 — relax landing_bonus thresholds to revive the dead component.
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
    left_contact  = next_obs[6]
    right_contact = next_obs[7]

    # ── distance to pad ──
    dist_cur  = (x_cur  ** 2 + y_cur  ** 2) ** 0.5
    dist_next = (x_next ** 2 + y_next ** 2) ** 0.5

    # ── weights / thresholds ──
    w_prox = 50.0
    w_vel  = 0.15
    w_ang  = 5.0
    proximity_threshold = 1.0

    w_land = 20.0                 # reduced from 80.0

    # ── 1. Proximity delta ──
    proximity_delta = w_prox * (dist_cur - dist_next)

    # ── 2. Velocity danger ──
    speed_sq = vx_cur ** 2 + vy_cur ** 2
    velocity_danger = -w_vel * speed_sq / (dist_cur + proximity_threshold)

    # ── 3. Orientation penalty ──
    orientation_penalty = -w_ang * (angle_cur ** 2 + angvel_cur ** 2)

    # ── 4. Soft landing bonus (relaxed) ──
    contact = max(left_contact, right_contact)

    # distance factor: slower decay (sigma 1.0 instead of 0.3)
    dist_factor = 2.718281828 ** (-dist_next / 1.0)

    # velocity and angle factors: wider linear ramp (cutoff 0.5 instead of 0.3)
    yvel_factor = max(0.0, 1.0 - abs(vy_cur) / 0.5)
    xvel_factor = max(0.0, 1.0 - abs(vx_cur) / 0.5)
    angle_factor = max(0.0, 1.0 - abs(angle_cur) / 0.5)

    landing_bonus = w_land * contact * dist_factor * yvel_factor * xvel_factor * angle_factor

    # ── Total reward ──
    total_reward = proximity_delta + velocity_danger + orientation_penalty + landing_bonus

    components = {
        "proximity_delta": proximity_delta,
        "velocity_danger": velocity_danger,
        "orientation_penalty": orientation_penalty,
        "landing_bonus": landing_bonus,
    }
    return float(total_reward), components