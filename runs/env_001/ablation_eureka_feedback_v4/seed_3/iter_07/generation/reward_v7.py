def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 当前状态
    x, y = obs[0], obs[1]
    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    nangle = next_obs[4]
    nleft, nright = next_obs[6], next_obs[7]

    # 1. Proximity: 向目标移动的距离减少量
    dist_curr = (x ** 2 + y ** 2) ** 0.5
    dist_next = (nx ** 2 + ny ** 2) ** 0.5
    progress = max(0.0, dist_curr - dist_next)

    # 2. Landing quality: 基于即时状态的着陆姿态、速度与接触奖励
    if ny < 0.3:
        angle_penalty = -0.2 * abs(nangle)
        vel_penalty = -0.1 * max(0.0, -nvy - 0.2)
        contact_reward = 1.0 * (nleft + nright)
        landing_quality = angle_penalty + vel_penalty + contact_reward
    else:
        landing_quality = -0.05 * abs(nangle)

    # 3. Energy penalty: 惩罚不必要的引擎使用
    energy_penalty = -0.01 if action != 0 else 0.0

    total_reward = progress + landing_quality + energy_penalty

    components = {
        "proximity": progress,
        "landing_quality": landing_quality,
        "energy_penalty": energy_penalty
    }
    return float(total_reward), components