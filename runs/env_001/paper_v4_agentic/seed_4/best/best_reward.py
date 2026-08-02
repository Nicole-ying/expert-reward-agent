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
    w_proximity = 10.0   # 大幅提升 soft_landing 继承者的权重

    th_angle  = 0.5    # radians, about 30 degrees
    th_vel    = 1.0    # speed magnitude
    th_angvel = 2.0    # rad/s
    th_dist   = 0.5    # proximity to target

    gate_min = 0.1     # for progress gate
    gate_min_stab = 0.2  # for stability factors to prevent collapse

    # ------------------- 1. progress signal (distance delta) -------------------
    delta_dist = max(0.0, dist - next_dist)

    gate_angle  = max(gate_min, 1.0 - abs(angle)  / th_angle)
    gate_vel    = max(gate_min, 1.0 - vel_abs      / th_vel)
    gate_angvel = max(gate_min, 1.0 - abs(angvel)  / th_angvel)
    gate = gate_angle * gate_vel * gate_angvel

    progress_gated = w_progress * delta_dist * gate

    # ------------------- 2. proximity + stability reward (replaces soft_landing) -------------------
    # proximity: how close to target (0,0)
    prox_factor = max(0.0, 1.0 - next_dist / th_dist)

    # stability after the step
    a_stab  = max(gate_min_stab, 1.0 - abs(n_angle)  / th_angle)
    v_stab  = max(gate_min_stab, 1.0 - next_vel_abs   / th_vel)
    av_stab = max(gate_min_stab, 1.0 - abs(n_angvel)  / th_angvel)
    stab = a_stab * v_stab * av_stab

    # contact gives a 1.5x multiplier, but is not required
    contact_flag = 1.0 if (n_left + n_right) >= 1.0 else 0.0
    contact_mult = 1.0 + 0.5 * contact_flag

    proximity_stability_reward = w_proximity * prox_factor * stab * contact_mult

    # ------------------- total reward -------------------
    total_reward = progress_gated + proximity_stability_reward

    components = {
        'progress_gated':   progress_gated,
        'proximity_stability': proximity_stability_reward
    }

    return float(total_reward), components