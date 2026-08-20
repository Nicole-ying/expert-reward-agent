def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 目标位置 (0,0)，obs 中 x,y 是相对于目标平台的坐标
    target_pos = (0.0, 0.0)

    # 1. 主学习信号：基于距离变化的进度奖励（保持不变）
    dist = ((obs[0] - target_pos[0]) ** 2 + (obs[1] - target_pos[1]) ** 2) ** 0.5
    next_dist = ((next_obs[0] - target_pos[0]) ** 2 + (next_obs[1] - target_pos[1]) ** 2) ** 0.5
    progress_reward = dist - next_dist

    # 2. 稳定性惩罚（保持不变）
    vel_x = abs(next_obs[2])
    vel_y = abs(next_obs[3])
    angle = abs(next_obs[4])
    ang_vel = abs(next_obs[5])
    stability_penalty = -0.001 * (vel_x + vel_y) - 0.001 * angle - 0.001 * ang_vel

    # 3. 连续软着陆 proxy — 从乘积改为接触门控加权和
    # 原因：乘积形态下 ∂(a·b·c)/∂a = b·c，任意因子小时所有梯度坍缩。
    # 和形态下每个姿势因子有独立梯度，contact_factor 作为全局门控保留 AND 约束。
    # 系数从 0.05 降至 0.01 补偿和形式天然更大的量级（三个 ~0.5 因子之和 ≈1.5，而乘积 ≈0.12）。
    dist_gate = 1.0 / (1.0 + 100.0 * next_dist ** 2)
    speed_factor = 1.0 / (1.0 + 5.0 * (vel_x + vel_y))
    angle_factor = 1.0 / (1.0 + 10.0 * angle)
    contact_factor = (next_obs[6] + next_obs[7]) / 2.0

    soft_landing_proxy = 0.01 * contact_factor * (dist_gate + speed_factor + angle_factor)

    total_reward = progress_reward + stability_penalty + soft_landing_proxy

    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing_proxy,
        "total_reward": total_reward
    }
    return float(total_reward), components