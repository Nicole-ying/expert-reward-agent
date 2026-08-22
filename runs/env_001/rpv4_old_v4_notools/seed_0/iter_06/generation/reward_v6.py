def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # -------------------------------
    # Hyperparameters
    # -------------------------------
    w_progress = 1.0          # weight for distance-reduction reward
    c_angle = 2.0             # health gate angle sensitivity
    c_vel = 1.0               # health gate speed sensitivity
    w_land = 2.0              # weight for continuous landing quality reward
    y_threshold = 5.0         # height below which landing quality activates
    speed_scale = 1.0         # speed penalty temperature
    angle_scale = 0.5         # angle penalty temperature
    contact_bonus = 10.0      # bonus for stable two-leg contact
    w_action = 0.01           # weight for action sparsity penalty

    # -------------------------------
    # 1. Goal-reaching progress (distance reduction with health gate)
    # -------------------------------
    dist_cur = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_raw = dist_cur - dist_next  # positive when getting closer

    health_penalty = c_angle * abs(obs[4]) + c_vel * (abs(obs[2]) + abs(obs[3]))
    health_gate = 1.0 / (1.0 + health_penalty)

    progress_reward = w_progress * progress_raw * health_gate

    # -------------------------------
    # 2. Continuous landing quality (height, speed, angle)
    # -------------------------------
    y = next_obs[1]                          # altitude (assuming ground at 0)
    height_factor = max(0.0, 1.0 - y / y_threshold)

    speed_total = abs(next_obs[2]) + abs(next_obs[3])
    angle_abs = abs(next_obs[4])

    # quality: high when low, slow, and upright
    quality = (
        height_factor *
        (2.718281828 ** (-speed_total / speed_scale)) *
        (2.718281828 ** (-angle_abs / angle_scale))
    )
    landing_reward = w_land * quality

    # bonus for actual stable two-leg contact
    both_legs_contact = (next_obs[6] > 0.5 and next_obs[7] > 0.5)
    if both_legs_contact:
        landing_reward += contact_bonus

    # -------------------------------
    # 3. Action sparsity penalty
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