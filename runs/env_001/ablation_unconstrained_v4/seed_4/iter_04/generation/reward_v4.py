def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- hyperparameters ----------
    w_prox = 12.0           # distance progress weight
    w_contact = 20.0        # contact bonus weight
    k_vel_pen = 0.8         # velocity penalty coefficient
    k_ang_pen = 1.0         # angle penalty coefficient
    v_threshold = 1.2       # safe speed threshold
    a_threshold = 0.3       # safe upright angle threshold (rad)

    # ---------- 1. distance progress ----------
    prev_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress = prev_dist - next_dist
    progress_reward = w_prox * progress

    # ---------- 2. contact bonus ----------
    contact = next_obs[6] * next_obs[7]   # 1.0 only when both legs touch
    contact_reward = w_contact * contact

    # ---------- 3. speed and angle penalties (hinge) ----------
    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    angle = abs(next_obs[4])
    vel_penalty = k_vel_pen * max(0.0, speed - v_threshold)
    ang_penalty = k_ang_pen * max(0.0, angle - a_threshold)

    # ---------- total ----------
    total_reward = progress_reward + contact_reward - vel_penalty - ang_penalty

    components = {
        'progress': progress_reward,
        'contact_reward': contact_reward,
        'vel_penalty': vel_penalty,
        'ang_penalty': ang_penalty,
    }
    return float(total_reward), components