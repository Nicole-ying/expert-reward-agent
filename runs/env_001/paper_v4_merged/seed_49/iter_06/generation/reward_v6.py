def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observations
    x, y = obs[0], obs[1]
    next_x, next_y = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]

    # ---------- 1) Main progress: distance reduction ----------
    dist = (x ** 2 + y ** 2) ** 0.5
    next_dist = (next_x ** 2 + next_y ** 2) ** 0.5
    delta_dist = dist - next_dist
    progress = 1.0 * delta_dist

    # ---------- 2) Overspeed penalty (replaces inactive angle/angvel) ----------
    speed = (vx ** 2 + vy ** 2) ** 0.5
    safe_speed = 0.5
    overspeed = max(0.0, speed - safe_speed)
    overspeed_penalty = -0.005 * overspeed

    # ---------- Total reward ----------
    total_reward = progress + overspeed_penalty

    components = {
        'progress': progress,
        'overspeed_penalty': overspeed_penalty
    }
    return float(total_reward), components