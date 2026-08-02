def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # observation indices: 0:x, 1:y, 2:vx, 3:vy, 4:angle, 5:ang_vel, 6:left_contact, 7:right_contact
    x, y = obs[0], obs[1]
    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    n_angle = next_obs[4]
    n_ang_vel = next_obs[5]

    # ---------- 1. progress towards origin (distance decrease) ----------
    dist_old = (x**2 + y**2) ** 0.5
    dist_new = (nx**2 + ny**2) ** 0.5
    progress = dist_old - dist_new
    w_progress = 10.0

    # ---------- 2. attitude gate: suppress progress when angle is dangerous ----------
    abs_angle = abs(n_angle)
    angle_gate = 1.0 - 0.8 * (2.718281828 ** (20.0 * (abs_angle - 0.15)) / 
                              (1.0 + 2.718281828 ** (20.0 * (abs_angle - 0.15))))

    # ---------- 3. lateral position penalty (encourage centering) ----------
    w_lat_pos = 0.15
    lateral_pos_penalty = -w_lat_pos * (nx ** 2)

    # ---------- 4. angular velocity penalty (smooth rotation) ----------
    w_angvel = 0.05
    angvel_penalty = -w_angvel * (n_ang_vel ** 2)

    # ---------- 5. continuous landing proxy (geometric mean) ----------
    k_y = 1.0
    k_vy = 10.0
    k_ang = 20.0
    f_y   = 1.0 / (1.0 + k_y   * abs(ny))
    f_vy  = 1.0 / (1.0 + k_vy  * abs(nvy))
    f_ang = 1.0 / (1.0 + k_ang * abs(n_angle))
    landing_factor = (f_y * f_vy * f_ang) ** (1.0 / 3.0)
    w_land = 2.0
    landing_reward = w_land * landing_factor

    # ---------- combine ----------
    total_reward = (w_progress * progress * angle_gate
                    + lateral_pos_penalty
                    + angvel_penalty
                    + landing_reward)

    components = {
        "progress_gated": w_progress * progress * angle_gate,
        "lateral_pos_penalty": lateral_pos_penalty,
        "angvel_penalty": angvel_penalty,
        "landing_reward": landing_reward
    }
    return float(total_reward), components