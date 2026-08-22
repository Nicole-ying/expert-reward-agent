def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observations
    x_pos, y_pos = obs[0], obs[1]
    x_vel, y_vel = obs[2], obs[3]
    angle = obs[4]
    ang_vel = obs[5]

    next_x, next_y = next_obs[0], next_obs[1]
    next_x_vel, next_y_vel = next_obs[2], next_obs[3]
    next_angle = next_obs[4]
    next_left = next_obs[6]
    next_right = next_obs[7]

    # 1. Soft landing proxy reward (main learning signal)
    landing_reward = 0.0
    if next_left > 0.5 and next_right > 0.5:
        pos_factor = 2.718281828 ** (-(next_x ** 2) / (2 * 0.0025))
        speed_n = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
        spd_factor = 2.718281828 ** (-(speed_n ** 2) / (2 * 0.04))
        ang_n = abs(next_angle)
        ang_factor = 2.718281828 ** (-(ang_n ** 2) / (2 * 0.01))
        landing_reward = 10.0 * pos_factor * spd_factor * ang_factor

    # 2. Progress reward: reduction in distance to target
    dist_now = (x_pos ** 2 + y_pos ** 2) ** 0.5
    dist_next = (next_x ** 2 + next_y ** 2) ** 0.5
    delta_dist = dist_now - dist_next

    near_target = dist_now < 0.5
    gate = 1.0
    if near_target:
        gate = 1.0 / (1.0 + 10.0 * (y_vel ** 2) + 5.0 * (angle ** 2))
    progress_reward = delta_dist * gate

    # 3. Action efficiency penalty
    action_cost = -0.01 if action != 0 else 0.0

    # 4. Safety penalty (replaces boundary_penalty)
    # Penalise dangerous descent: too fast downward speed when close to ground,
    # amplified by body tilt.
    height_limit = 0.3
    v_limit = 0.2          # safe downward speed threshold (negative means down, so -y_vel positive)
    proximity = max(0.0, 1.0 - y_pos / height_limit)  # [0,1] when y_pos < 0.3
    danger_speed = max(0.0, -y_vel - v_limit)         # >0 when downward speed exceeds limit
    attitude = 1.0 + 2.0 * abs(angle)                 # tilt penalty multiplier
    safety_penalty = -0.2 * danger_speed * proximity * attitude

    # 5. Light angle/angular-velocity penalty
    angle_penalty = -0.01 * abs(angle) - 0.001 * abs(ang_vel)

    total_reward = (landing_reward + progress_reward +
                    action_cost + safety_penalty + angle_penalty)

    components = {
        "landing_soft_reward": landing_reward,
        "progress": progress_reward,
        "action_cost": action_cost,
        "safety_penalty": safety_penalty,
        "angle_penalty": angle_penalty
    }
    return float(total_reward), components