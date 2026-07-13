def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- hyperparameters ----------
    w_prox = 25.0               # distance progress weight (increased)
    w_contact_base = 2.0        # low base contact reward
    w_contact_quality = 30.0    # quality bonus for good landing
    v_soft = 0.5                # target speed for quality (soft limit)
    a_soft = 0.25               # target angle for quality (soft limit)
    k_vel_pen = 2.0             # velocity penalty coefficient
    k_ang_pen = 2.0             # angle penalty coefficient
    v_threshold = 0.6           # safe speed threshold (lowered)
    a_threshold = 0.2           # safe angle threshold (rad)
    fuel_cost = 0.5             # fuel penalty per engine use

    # ---------- 1. distance progress ----------
    prev_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress = prev_dist - next_dist
    progress_reward = w_prox * progress

    # ---------- 2. contact reward (base + quality) ----------
    contact = next_obs[6] * next_obs[7]   # 1.0 when both legs touch
    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    angle = abs(next_obs[4])
    quality_speed = max(0.0, 1.0 - speed / v_soft)
    quality_angle = max(0.0, 1.0 - angle / a_soft)
    quality = (quality_speed * quality_angle) ** 0.5
    contact_reward = w_contact_base * contact + w_contact_quality * contact * quality

    # ---------- 3. speed and angle penalties (hinge) ----------
    vel_penalty = k_vel_pen * max(0.0, speed - v_threshold)
    ang_penalty = k_ang_pen * max(0.0, angle - a_threshold)

    # ---------- 4. fuel penalty ----------
    fuel_penalty = fuel_cost * float(action != 0)

    # ---------- total ----------
    total_reward = progress_reward + contact_reward - vel_penalty - ang_penalty - fuel_penalty

    components = {
        'progress': progress_reward,
        'contact_reward': contact_reward,
        'vel_penalty': vel_penalty,
        'ang_penalty': ang_penalty,
        'fuel_penalty': fuel_penalty,
    }
    return float(total_reward), components