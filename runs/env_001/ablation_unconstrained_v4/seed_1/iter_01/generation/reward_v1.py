def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- extract current and next state ----
    x, y = obs[0], obs[1]
    vx, vy = obs[2], obs[3]
    angle = obs[4]
    angvel = obs[5]
    left_contact = obs[6]
    right_contact = obs[7]

    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    n_angle = next_obs[4]
    n_angvel = next_obs[5]
    n_left_contact = next_obs[6]
    n_right_contact = next_obs[7]

    # Component A: distance progress (main learning signal)
    prev_dist = (x**2 + y**2) ** 0.5
    next_dist = (nx**2 + ny**2) ** 0.5
    progress = prev_dist - next_dist
    w_progress = 1.0
    c_progress = w_progress * progress

    # Component B: soft landing bonus (task-completion proxy)
    dist_land = (nx**2 + ny**2) ** 0.5
    speed_land = (nvx**2 + nvy**2) ** 0.5
    contact_flag = n_left_contact * n_right_contact   # 1.0 if both legs touch

    prox_score = 2.718281828 ** (-dist_land**2 / 0.1)
    vel_score  = 2.718281828 ** (-(nvx**2 + nvy**2) / 0.1)
    angle_score = 2.718281828 ** (-(n_angle**2) / 0.01)
    angvel_score = 2.718281828 ** (-(n_angvel**2) / 0.01)

    landing_bonus = contact_flag * prox_score * vel_score * angle_score * angvel_score
    w_landing = 10.0
    c_landing = w_landing * landing_bonus

    # Component C: posture penalty (stability constraint)
    w_angle_pen = 0.1
    w_angvel_pen = 0.001
    c_posture = -w_angle_pen * (n_angle**2) - w_angvel_pen * (n_angvel**2)

    total_reward = c_progress + c_landing + c_posture
    components = {
        'distance_progress': c_progress,
        'soft_landing_bonus': c_landing,
        'posture_penalty': c_posture
    }
    return float(total_reward), components