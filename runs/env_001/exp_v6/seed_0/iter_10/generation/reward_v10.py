def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 目标位置 (0,0)，obs 中 x,y 是相对于目标平台的坐标
    target_pos = (0.0, 0.0)

    # 1. 主学习信号：进度奖励 — scale=5.0（骨架推荐 5~20，取保守下限）
    #    之前 scale=1.0 导致 progress_reward 均值仅 0.003，被 proxy 淹没。
    dist = ((obs[0] - target_pos[0]) ** 2 + (obs[1] - target_pos[1]) ** 2) ** 0.5
    next_dist = ((next_obs[0] - target_pos[0]) ** 2 + (next_obs[1] - target_pos[1]) ** 2) ** 0.5
    progress_reward = 5.0 * (dist - next_dist)

    # 2. 稳定性惩罚（保持不变）
    vel_x = abs(next_obs[2])
    vel_y = abs(next_obs[3])
    angle = abs(next_obs[4])
    ang_vel = abs(next_obs[5])
    stability_penalty = -0.001 * (vel_x + vel_y) - 0.001 * angle - 0.001 * ang_vel

    # 3. 连续软着陆 proxy — 收紧距离门控 + 降低系数 + 要求双腿同时触地
    #    距离门控从 100→500：dist=0.05→0.44, dist=0.1→0.17, dist=0.2→0.05
    #    只在极靠近目标时激活，杜绝"附近徘徊收割 proxy"。
    #    接触因子用 min 而非 mean：要求双腿同时接地才给满信号。
    #    系数从 0.01→0.003，配合 progress 5x 放大后保持总 reward 合理。
    dist_gate = 1.0 / (1.0 + 500.0 * next_dist ** 2)
    speed_factor = 1.0 / (1.0 + 5.0 * (vel_x + vel_y))
    angle_factor = 1.0 / (1.0 + 10.0 * angle)
    # 要求双腿同时触地：min 取代 mean，单腿悬空时信号大幅衰减
    contact_factor = min(next_obs[6], next_obs[7])

    soft_landing_proxy = 0.003 * dist_gate * speed_factor * angle_factor * contact_factor

    total_reward = progress_reward + stability_penalty + soft_landing_proxy

    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing_proxy,
        "total_reward": total_reward
    }
    return float(total_reward), components