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
    next_vel_abs  = (nvx**2 + nvy**2) ** 0.5

    # ------------------- thresholds & weights -------------------
    w_progress = 8.0
    w_delta = 20.0
    w_landing = 200.0
    w_fuel = 0.2

    th_angle  = 0.5
    th_vel    = 1.0
    th_angvel = 2.0

    # ------------------- 1. velocity-toward-target progress signal -------------------
    dir_x = -nx / (next_dist + 1e-6)
    dir_y = -ny / (next_dist + 1e-6)
    toward_speed = nvx * dir_x + nvy * dir_y

    gate_min = 0.1
    gate_angle  = max(gate_min, 1.0 - abs(n_angle)  / th_angle)
    gate_vel    = max(gate_min, 1.0 - next_vel_abs   / th_vel)
    gate_angvel = max(gate_min, 1.0 - abs(n_angvel)  / th_angvel)
    gate = gate_angle * gate_vel * gate_angvel

    progress_gated = w_progress * max(0.0, toward_speed) * gate

    # ------------------- 2. landing progress (replaces proximity_stability) -------------------
    # Check plausible landing condition
    landing_cond = (next_dist < 0.15 and
                    (n_left + n_right) >= 1.0 and
                    abs(n_angle) < 0.3 and
                    next_vel_abs < 0.5 and
                    abs(n_angvel) < 0.5)

    if landing_cond:
        landing_progress = w_landing
    else:
        # stability gate without floor – zero if unstable
        a_stab  = max(0.0, 1.0 - abs(n_angle)  / th_angle)
        v_stab  = max(0.0, 1.0 - next_vel_abs   / th_vel)
        av_stab = max(0.0, 1.0 - abs(n_angvel)  / th_angvel)
        stab = a_stab * v_stab * av_stab

        distance_reduction = max(0.0, dist - next_dist)
        landing_progress = w_delta * distance_reduction * stab

    # ------------------- 3. fuel penalty -------------------
    engine_on = 1.0 if action != 0 else 0.0
    fuel_penalty = -w_fuel * engine_on

    # ------------------- total reward -------------------
    total_reward = progress_gated + landing_progress + fuel_penalty

    components = {
        'progress_gated':   progress_gated,
        'landing_progress': landing_progress,
        'fuel_penalty':     fuel_penalty
    }

    return float(total_reward), components