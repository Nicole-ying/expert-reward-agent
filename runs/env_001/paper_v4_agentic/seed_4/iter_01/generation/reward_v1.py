def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ------------------- unpack observations -------------------
    x,  y  = obs[0], obs[1]
    vx, vy = obs[2], obs[3]
    angle      = obs[4]
    angvel     = obs[5]
    left_leg   = obs[6]
    right_leg  = obs[7]

    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    n_angle  = next_obs[4]
    n_angvel = next_obs[5]
    n_left   = next_obs[6]
    n_right  = next_obs[7]

    # ------------------- helper quantities -------------------
    dist      = (x**2  + y**2)  ** 0.5
    next_dist = (nx**2 + ny**2) ** 0.5
    vel_abs       = (vx**2 + vy**2) ** 0.5
    next_vel_abs  = (nvx**2 + nvy**2) ** 0.5

    # ------------------- thresholds & weights -------------------
    w_progress = 1.0
    w_landing  = 2.0

    th_angle  = 0.5    # radians, about 30 degrees
    th_vel    = 1.0    # speed magnitude
    th_angvel = 2.0    # rad/s
    th_dist   = 0.5    # proximity to target for landing bonus

    gate_min = 0.1  # floor for each individual gate

    # ------------------- 1. progress signal (distance delta) -------------------
    # only reward moving closer, no penalty for moving away
    delta_dist = max(0.0, dist - next_dist)

    # ------------------- 2. soft health gate -------------------
    gate_angle  = max(gate_min, 1.0 - abs(angle)  / th_angle)
    gate_vel    = max(gate_min, 1.0 - vel_abs      / th_vel)
    gate_angvel = max(gate_min, 1.0 - abs(angvel)  / th_angvel)
    gate = gate_angle * gate_vel * gate_angvel

    progress_gated = w_progress * delta_dist * gate

    # ------------------- 3. soft landing proxy -------------------
    # contact: at least one leg touching in the next state
    contact_next = 1.0 if (n_left + n_right) >= 1.0 else 0.0

    # stability factors after the step
    factor_angle  = max(0.0, 1.0 - abs(n_angle)  / th_angle)
    factor_vel    = max(0.0, 1.0 - next_vel_abs   / th_vel)
    factor_angvel = max(0.0, 1.0 - abs(n_angvel)  / th_angvel)
    factor_dist   = max(0.0, 1.0 - next_dist       / th_dist)

    landing_score = contact_next * factor_angle * factor_vel * factor_angvel * factor_dist
    landing_reward = w_landing * landing_score

    # ------------------- total reward -------------------
    total_reward = progress_gated + landing_reward

    components = {
        'progress_gated': progress_gated,
        'soft_landing':    landing_reward
    }

    return float(total_reward), components