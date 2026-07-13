def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- hyperparameters ----------
    w_prox   = 5.0   # distance reduction weight (increased to dominate)
    w_vel    = 0.3   # landing velocity penalty weight
    w_angle  = 0.2   # body angle penalty weight
    w_angvel = 0.1   # angular velocity penalty weight
    w_xcenter = 0.5  # penalty for horizontal offset from center
    w_hspeed  = 0.2  # penalty for horizontal speed

    k_gate_y = 20.0  # steepness of height gate

    # ---------- 1. progress: distance reduction ----------
    prev_dist = (obs[0]**2 + obs[1]**2) ** 0.5
    next_dist = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    progress  = prev_dist - next_dist
    progress_reward = w_prox * progress

    # ---------- 2. soft landing velocity penalty ----------
    y_clipped = max(0.0, next_obs[1])   # y positive above pad
    gate_vel  = 1.0 / (1.0 + k_gate_y * y_clipped)
    speed     = (next_obs[2]**2 + next_obs[3]**2) ** 0.5
    landing_vel_penalty = -w_vel * speed * gate_vel

    # ---------- 3. upright stability penalty ----------
    upright_penalty = -w_angle * abs(next_obs[4]) - w_angvel * abs(next_obs[5])

    # ---------- 4. horizontal offset penalty ----------
    x_center_penalty = -w_xcenter * abs(next_obs[0])

    # ---------- 5. horizontal speed penalty ----------
    hspeed_penalty = -w_hspeed * abs(next_obs[2])

    # ---------- total reward ----------
    total_reward = (progress_reward +
                    landing_vel_penalty +
                    upright_penalty +
                    x_center_penalty +
                    hspeed_penalty)

    components = {
        'progress': progress_reward,
        'landing_vel_penalty': landing_vel_penalty,
        'upright_penalty': upright_penalty,
        'x_center_penalty': x_center_penalty,
        'hspeed_penalty': hspeed_penalty
    }
    return float(total_reward), components