def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Current observation
    x = float(obs[0])
    y = float(obs[1])

    # Next observation
    nx = float(next_obs[0])
    ny = float(next_obs[1])
    nvx = float(next_obs[2])
    nvy = float(next_obs[3])
    nangle = float(next_obs[4])
    left_contact = float(next_obs[6])
    right_contact = float(next_obs[7])

    # ---------- 1. Bounded distance-based progress ----------
    dist_obs = (x * x + y * y) ** 0.5
    dist_next = (nx * nx + ny * ny) ** 0.5
    dist_delta = dist_obs - dist_next   # positive = getting closer

    progress_pos = 0.5 * max(0.0, dist_delta)
    progress_neg = 0.05 * min(0.0, dist_delta)
    progress_shaping = progress_pos + progress_neg

    # ---------- 2. Landing speed gate ----------
    proximity_factor = max(0.0, 1.0 - dist_next / 0.5)
    speed_next = (nvx * nvx + nvy * nvy) ** 0.5
    speed_cost_input = speed_next * proximity_factor
    landing_speed_gate = 1.0 / (1.0 + 5.0 * speed_cost_input)

    shaped_progress = progress_shaping * landing_speed_gate

    # ---------- 3. Action cost ----------
    action_cost = -0.01 * (0.0 if action == 0 else 1.0)

    # ---------- 4. Landing contact bonus ----------
    contact_sum = left_contact + right_contact
    contact_factor = contact_sum / 2.0
    proximity = max(0.0, 1.0 - dist_next / 0.8)
    landing_contact_reward = 0.2 * contact_factor * proximity

    # ---------- 5. Angle hinge penalty (kept as is) ----------
    angle_abs = abs(nangle)
    angle_excess = max(0.0, angle_abs - 0.3)
    angle_hinge_penalty = -0.03 * angle_excess

    total_reward = shaped_progress + action_cost + landing_contact_reward + angle_hinge_penalty

    components = {
        "progress_shaping": progress_shaping,
        "shaped_progress": shaped_progress,
        "action_cost": action_cost,
        "landing_contact_reward": landing_contact_reward,
        "angle_hinge_penalty": angle_hinge_penalty
    }

    return float(total_reward), components