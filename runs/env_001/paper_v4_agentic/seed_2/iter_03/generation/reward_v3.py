def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v3 reward — add soft_landing_bonus using leg contact to signal successful touchdown.
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

    # ── distance to pad (target at 0, 0) ──
    dist_cur  = (x_cur  ** 2 + y_cur  ** 2) ** 0.5
    dist_next = (x_next ** 2 + y_next ** 2) ** 0.5

    # ── weights / thresholds ──
    w_prox = 50.0
    w_vel  = 0.15
    w_ang  = 5.0
    proximity_threshold = 1.0

    w_land = 80.0            # soft landing bonus weight

    # ── 1. Proximity delta (unchanged) ──
    proximity_delta = w_prox * (dist_cur - dist_next)

    # ── 2. Velocity danger (unchanged) ──
    speed_sq = vx_cur ** 2 + vy_cur ** 2
    velocity_danger = -w_vel * speed_sq / (dist_cur + proximity_threshold)

    # ── 3. Orientation penalty (unchanged) ──
    orientation_penalty = -w_ang * (angle_cur ** 2 + angvel_cur ** 2)

    # ── 4. Soft landing bonus (NEW) ──
    contact = max(left_contact, right_contact)  # 0.0 or 1.0

    # distance factor: exponential decay as distance to target increases
    dist_factor = 2.718281828 ** (-dist_next / 0.3)

    # velocity and angle factors: linear ramp from 1 at 0 to 0 at threshold 0.3
    yvel_factor = max(0.0, 1.0 - abs(vy_cur) / 0.3)
    xvel_factor = max(0.0, 1.0 - abs(vx_cur) / 0.3)
    angle_factor = max(0.0, 1.0 - abs(angle_cur) / 0.3)

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