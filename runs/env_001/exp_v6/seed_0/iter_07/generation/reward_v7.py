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

    # 3. 连续软着陆 proxy — 从二值改为 bounded 乘积，全程梯度 + 自动 anti-exploit
    # near_factor: 1/(1+10*dist), dist=0→1.0, dist=0.1→0.5, dist=0.5→0.17
    near_factor = 1.0 / (1.0 + 10.0 * next_dist)
    # speed_factor: 1/(1+5*speed), speed=0→1.0, speed=0.2→0.5, speed=0.5→0.29
    speed_factor = 1.0 / (1.0 + 5.0 * (vel_x + vel_y))
    # angle_factor: 1/(1+10*angle), angle=0→1.0, angle=0.1→0.5, angle=0.3→0.25
    angle_factor = 1.0 / (1.0 + 10.0 * angle)
    # contact_factor: 连续 0~1，两条腿平均接触
    contact_factor = (next_obs[6] + next_obs[7]) / 2.0

    soft_landing_proxy = 0.1 * near_factor * speed_factor * angle_factor * contact_factor

    total_reward = progress_reward + stability_penalty + soft_landing_proxy

    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing_proxy,
        "total_reward": total_reward
    }
    return float(total_reward), components