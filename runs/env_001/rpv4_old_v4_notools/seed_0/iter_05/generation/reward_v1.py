def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # -------------------------------
    # Hyperparameters
    # -------------------------------
    w_progress = 1.0
    c_angle = 2.0      # health gate sensitivity for body angle
    c_vel = 1.0        # health gate sensitivity for total speed
    w_land = 5.0
    angle_thresh = 0.5   # rad, quality for landing
    vel_thresh = 1.0     # m/s (or units), total speed quality for landing
    w_action = 0.01

    # -------------------------------
    # 1. Goal‑reaching progress (improvement_delta)
    # -------------------------------
    dist_cur = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_unscaled = dist_cur - dist_next  # positive when getting closer

    # -------------------------------
    # 2. Soft health gate (on current body state)
    # -------------------------------
    # Use body angle and total speed to scale progress
    health_penalty = c_angle * abs(obs[4]) + c_vel * (abs(obs[2]) + abs(obs[3]))
    health_gate = 1.0 / (1.0 + health_penalty)

    progress_reward = w_progress * progress_unscaled * health_gate

    # -------------------------------
    # 3. Landing quality (joint condition proxy)
    # -------------------------------
    both_legs_contact = 1.0 if (next_obs[6] > 0.5 and next_obs[7] > 0.5) else 0.0
    if both_legs_contact > 0.0:
        angle_factor = max(0.0, 1.0 - abs(next_obs[4]) / angle_thresh)
        speed_total = abs(next_obs[2]) + abs(next_obs[3])
        speed_factor = max(0.0, 1.0 - speed_total / vel_thresh)
        landing_quality = angle_factor * speed_factor
    else:
        landing_quality = 0.0

    landing_reward = w_land * landing_quality

    # -------------------------------
    # 4. Action sparsity
    # -------------------------------
    action_penalty = -w_action * (1.0 if action != 0 else 0.0)

    # -------------------------------
    # Total reward
    # -------------------------------
    total_reward = progress_reward + landing_reward + action_penalty
    components = {
        "progress_reward": progress_reward,
        "landing_reward": landing_reward,
        "action_penalty": action_penalty
    }
    return float(total_reward), components