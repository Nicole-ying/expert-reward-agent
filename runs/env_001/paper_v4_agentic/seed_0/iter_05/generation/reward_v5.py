def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation variables
    x, y = obs[0], obs[1]
    x_v, y_v = obs[2], obs[3]
    angle = obs[4]
    ang_v = obs[5]

    nx, ny = next_obs[0], next_obs[1]
    nx_v, ny_v = next_obs[2], next_obs[3]
    n_angle = next_obs[4]
    n_ang_v = next_obs[5]
    n_left = next_obs[6]
    n_right = next_obs[7]

    # ---------- 1. Progress reward: moving toward the landing pad (0,0) ----------
    dist_curr = (x**2 + y**2) ** 0.5
    dist_next = (nx**2 + ny**2) ** 0.5
    progress = dist_curr - dist_next
    progress_reward = 1.0 * progress

    # ---------- 2. Landing contact bonus (continuous, gated by distance) ----------
    contact = (n_left + n_right) / 2.0
    gate_contact = 1.0 / (1.0 + 3.0 * dist_next)
    landing_contact_bonus = 2.0 * contact * gate_contact

    # ---------- 3. Landing softness / safety penalty (reduced coefficients) ----------
    v_limit = 0.5
    vx_pen = max(0.0, abs(nx_v) - v_limit)
    vy_pen = max(0.0, abs(ny_v) - v_limit)
    vel_pen = vx_pen + vy_pen

    ang_limit = 1.0
    ang_pen = max(0.0, abs(n_ang_v) - ang_limit)

    tilt_pen = abs(n_angle)

    gate_safety = 1.0 / (1.0 + 5.0 * dist_next)
    landing_safety_penalty = (0.03 * vel_pen + 0.02 * ang_pen + 0.03 * tilt_pen) * gate_safety

    # ---------- Total reward ----------
    total_reward = progress_reward + landing_contact_bonus - landing_safety_penalty

    components = {
        "progress_reward": float(progress_reward),
        "landing_contact_bonus": float(landing_contact_bonus),
        "landing_safety_penalty": float(landing_safety_penalty)
    }
    return float(total_reward), components