def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observations
    prev_x, prev_y = obs[0], obs[1]
    prev_vx, prev_vy = obs[2], obs[3]

    next_x, next_y = next_obs[0], next_obs[1]
    next_vx, next_vy = next_obs[2], next_obs[3]
    next_angle = next_obs[4]
    next_left_contact = next_obs[6]
    next_right_contact = next_obs[7]

    # Distance to target pad (0,0)
    dist_prev = (prev_x**2 + prev_y**2) ** 0.5
    dist_next = (next_x**2 + next_y**2) ** 0.5

    # 1. Goal distance progress (improvement delta)
    w_progress = 5.0
    progress = w_progress * (dist_prev - dist_next)

    # 2. Soft landing (proximity-gated velocity penalty)  <-- modified weight from 1.0 to 2.0
    w_vel = 2.0
    k_dist = 20.0
    proximity_gate = 1.0 / (1.0 + k_dist * dist_next**2)
    vel_penalty = -w_vel * (next_vx**2 + next_vy**2) * proximity_gate

    # 3. Orientation penalty (quadratic on tilt)
    w_angle = 0.5
    angle_penalty = -w_angle * (next_angle**2)

    # 4. Landing contact success (continuous proximity-gated quality)
    w_success = 5.0
    k_dist_success = 5.0
    k_speed = 2.0
    k_angle_success = 5.0
    legs_on_ground = (next_left_contact >= 0.5) and (next_right_contact >= 0.5)
    contact_flag = 1.0 if legs_on_ground else 0.0
    dist = (next_x**2 + next_y**2) ** 0.5
    speed = (next_vx**2 + next_vy**2) ** 0.5
    angle_abs = abs(next_angle)
    quality = (2.718281828 ** (-k_dist_success * dist)) * \
              (2.718281828 ** (-k_speed * speed)) * \
              (2.718281828 ** (-k_angle_success * angle_abs))
    contact_bonus = w_success * contact_flag * quality

    # Aggregate
    total_reward = progress + vel_penalty + angle_penalty + contact_bonus

    components = {
        "goal_distance_progress": progress,
        "soft_landing": vel_penalty,
        "orientation_penalty": angle_penalty,
        "landing_contact_success": contact_bonus
    }

    return float(total_reward), components