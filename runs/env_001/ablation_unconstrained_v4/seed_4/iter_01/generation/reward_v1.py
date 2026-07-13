def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- hyperparameters ----------
    w_prox   = 1.0   # distance reduction weight
    w_vel    = 0.3   # landing velocity penalty weight
    w_angle  = 0.2   # body angle penalty weight
    w_angvel = 0.1   # angular velocity penalty weight
    w_bonus  = 0.5   # stable‑stand bonus weight

    k_gate_y = 20.0  # steepness of height gate
    k_fx     = 5.0   # factor scaling for horizontal distance
    k_fy     = 5.0   # factor scaling for vertical height
    k_fspeed = 2.0   # factor scaling for speed
    k_fangle = 2.0   # factor scaling for body angle

    # ---------- 1. progress: distance reduction ----------
    prev_dist = (obs[0]**2 + obs[1]**2) ** 0.5
    next_dist = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    progress  = prev_dist - next_dist
    progress_reward = w_prox * progress

    # ---------- 2. soft landing velocity penalty ----------
    # gate = 1 when y is near zero (height close to pad), decays when high
    y_clipped = max(0.0, next_obs[1])   # y positive above pad
    gate_vel  = 1.0 / (1.0 + k_gate_y * y_clipped)
    speed     = (next_obs[2]**2 + next_obs[3]**2) ** 0.5
    landing_vel_penalty = -w_vel * speed * gate_vel

    # ---------- 3. upright stability penalty ----------
    upright_penalty = -w_angle * abs(next_obs[4]) - w_angvel * abs(next_obs[5])

    # ---------- 4. stable‑stand bonus (joint condition proxy) ----------
    # each factor ∈ [0,1]; geometric mean avoids collapse
    factor_x       = 1.0 / (1.0 + k_fx * abs(next_obs[0]))
    factor_y       = 1.0 / (1.0 + k_fy * max(0.0, next_obs[1]))
    factor_speed   = 1.0 / (1.0 + k_fspeed * speed)
    factor_angle   = 1.0 / (1.0 + k_fangle * abs(next_obs[4]))
    factor_contact = next_obs[6] * next_obs[7]  # 1 only when both legs touch

    prod = factor_x * factor_y * factor_speed * factor_angle * factor_contact
    stable_bonus = w_bonus * (prod ** 0.2) if prod > 0.0 else 0.0

    # ---------- total reward ----------
    total_reward = progress_reward + landing_vel_penalty + upright_penalty + stable_bonus

    components = {
        'progress': progress_reward,
        'landing_vel_penalty': landing_vel_penalty,
        'upright_penalty': upright_penalty,
        'stable_bonus': stable_bonus
    }
    return float(total_reward), components