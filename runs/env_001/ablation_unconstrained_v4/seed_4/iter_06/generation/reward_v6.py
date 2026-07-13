def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- hyperparameters ----------
    w_prox = 10.0               # progress weight
    w_contact_base = 2.0        # constant contact bonus
    w_contact_quality = 30.0    # quality multiplier for soft landing
    v_max = 1.0                 # max safe speed for progress gate
    a_max = 0.5                 # max safe angle (rad) for progress gate
    v_soft = 0.5                # target speed for contact quality
    a_soft = 0.25               # target angle for contact quality

    # ---------- 1. distance progress (gate modulation) ----------
    prev_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress = prev_dist - next_dist

    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    angle = abs(next_obs[4])

    # health gates for progress: low speed & upright attitude are required to earn progress
    gate_speed = max(0.0, 1.0 - speed / v_max)
    gate_angle = max(0.0, 1.0 - angle / a_max)
    effective_progress = w_prox * progress * gate_speed * gate_angle

    # ---------- 2. contact reward (base + quality) ----------
    contact = next_obs[6] * next_obs[7]   # 1.0 only when both legs touch
    quality_speed = max(0.0, 1.0 - speed / v_soft)
    quality_angle = max(0.0, 1.0 - angle / a_soft)
    quality = (quality_speed * quality_angle) ** 0.5
    contact_reward = w_contact_base * contact + w_contact_quality * contact * quality

    # ---------- total (no independent penalties) ----------
    total_reward = effective_progress + contact_reward

    components = {
        'progress': effective_progress,
        'contact_reward': contact_reward,
    }
    return float(total_reward), components