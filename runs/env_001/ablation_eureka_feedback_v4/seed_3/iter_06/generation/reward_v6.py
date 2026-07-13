def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 当前状态
    x, y = obs[0], obs[1]
    vx, vy = obs[2], obs[3]
    angle = obs[4]
    left, right = obs[6], obs[7]

    # 下一状态
    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    nangle = next_obs[4]
    nleft, nright = next_obs[6], next_obs[7]

    # 1. Proximity: 向目标移动的距离减少量
    dist_curr = (x ** 2 + y ** 2) ** 0.5
    dist_next = (nx ** 2 + ny ** 2) ** 0.5
    progress = max(0.0, dist_curr - dist_next)

    # 2. Landing improvement: 奖励姿态和速度的改善（平方减少）
    angle_sq = angle ** 2
    nangle_sq = nangle ** 2
    speed_sq = vx ** 2 + vy ** 2
    nspeed_sq = nvx ** 2 + nvy ** 2

    angle_improvement = max(0.0, angle_sq - nangle_sq)
    speed_improvement = max(0.0, speed_sq - nspeed_sq)
    landing_improvement = angle_improvement + speed_improvement

    # 3. Energy penalty
    energy_penalty = -0.01 if action != 0 else 0.0

    # 4. Terminal velocity penalty: 近地时向下速度过大
    vel_penalty = 0.0
    if ny < 0.1 and nvy < -0.3:
        vel_penalty = -1.0 * abs(nvy + 0.3)

    total_reward = progress + landing_improvement * 0.5 + energy_penalty + vel_penalty

    components = {
        "proximity": progress,
        "landing_improvement": landing_improvement * 0.5,
        "energy_penalty": energy_penalty,
        "terminal_velocity_penalty": vel_penalty
    }
    return float(total_reward), components