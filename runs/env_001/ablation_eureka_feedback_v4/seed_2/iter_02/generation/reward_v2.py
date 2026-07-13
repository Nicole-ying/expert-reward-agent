def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 当前状态
    x, y = obs[0], obs[1]
    # 下一状态
    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    n_angle = next_obs[4]
    n_lc = next_obs[6]
    n_rc = next_obs[7]

    # 距离
    dist_old = (x**2 + y**2)**0.5
    dist_new = (nx**2 + ny**2)**0.5

    # 1. 目标进展
    w_progress = 1.0
    progress = w_progress * (dist_old - dist_new)

    # 2. 软着陆：靠近目标时激活，鼓励低速、正姿态、双足接触
    landing_threshold = 0.5
    gate = max(0.0, 1.0 - dist_new / landing_threshold)

    w_contact = 0.5
    contact_signal = w_contact * (n_lc + n_rc)   # 连续梯度，最大 1.0

    safe_speed = 0.2
    w_speed = 0.5
    speed_mag = (nvx**2 + nvy**2)**0.5
    speed_penalty = w_speed * max(0.0, speed_mag - safe_speed)

    safe_angle = 0.1
    w_angle = 0.5
    angle_abs = abs(n_angle)
    angle_penalty = w_angle * max(0.0, angle_abs - safe_angle)

    soft_landing = gate * (contact_signal - speed_penalty - angle_penalty)

    # 3. 燃料效率
    w_fuel = 0.02
    fuel_cost = -w_fuel * float(action != 0)

    total_reward = progress + soft_landing + fuel_cost
    components = {
        "progress": progress,
        "soft_landing": soft_landing,
        "fuel_cost": fuel_cost
    }
    return float(total_reward), components