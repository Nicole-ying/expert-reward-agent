def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # observation indices: 0:x, 1:y, 2:vx, 3:vy, 4:angle, 5:ang_vel, 6:left_contact, 7:right_contact
    x, y = obs[0], obs[1]
    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    n_angle = next_obs[4]
    n_ang_vel = next_obs[5]
    n_lc = next_obs[6]  # left contact
    n_rc = next_obs[7]  # right contact

    # ---------- 1. progress towards origin (distance decrease) ----------
    dist_old = (x**2 + y**2) ** 0.5
    dist_new = (nx**2 + ny**2) ** 0.5
    progress = dist_old - dist_new
    w_progress = 30.0

    # ---------- 2. attitude gate: suppress progress when angle is dangerous ----------
    abs_angle = abs(n_angle)
    # sigmoid with steepness 12, center 0.30
    angle_gate = 1.0 - 0.8 * (2.718281828 ** (12.0 * (abs_angle - 0.30)) / 
                              (1.0 + 2.718281828 ** (12.0 * (abs_angle - 0.30))))

    # ---------- 3. lateral position penalty (encourage centering) ----------
    w_lat_pos = 0.08
    lateral_pos_penalty = -w_lat_pos * (nx ** 2)

    # ---------- 4. angular velocity penalty (smooth rotation) ----------
    w_angvel = 0.05
    angvel_penalty = -w_angvel * (n_ang_vel ** 2)

    # ---------- 5. contact-based landing proxy (exponent 0.5) ----------
    mean_contact = (n_lc + n_rc) / 2.0

    k_y = 10.0
    k_vy = 8.0
    k_ang = 15.0
    f_y   = 1.0 / (1.0 + k_y   * abs(ny))
    f_vy  = 1.0 / (1.0 + k_vy  * abs(nvy))
    f_ang = 1.0 / (1.0 + k_ang * abs(n_angle))

    contact_landing_factor = (mean_contact * f_y * f_vy * f_ang) ** 0.5
    w_contact_land = 5.0
    contact_landing_reward = w_contact_land * contact_landing_factor

    # ---------- 6. NEW: angle-terminal-proximity penalty ----------
    # Hinge penalty activates when angle approaches a dangerous region
    # (estimated termination threshold around 1.0 rad; start warning at 0.7)
    angle_term_threshold = 0.7
    w_angle_term = 1.5
    angle_term_penalty = -w_angle_term * (max(0.0, abs_angle - angle_term_threshold) ** 2)

    # ---------- combine ----------
    total_reward = (w_progress * progress * angle_gate
                    + lateral_pos_penalty
                    + angvel_penalty
                    + contact_landing_reward
                    + angle_term_penalty)

    components = {
        "progress_gated": w_progress * progress * angle_gate,
        "lateral_pos_penalty": lateral_pos_penalty,
        "angvel_penalty": angvel_penalty,
        "contact_landing_reward": contact_landing_reward,
        "angle_term_penalty": angle_term_penalty
    }
    return float(total_reward), components