def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取下一步状态
    nx, ny = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    omega = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 权重和阈值（distance_cost 与 stability_cost 保持不变）
    w_dist = 1.0
    w_vel = 0.1
    w_angle = 0.1
    w_angvel = 0.05

    # 组件1: 距离成本（保持不变）
    distance_cost = -w_dist * (nx ** 2 + ny ** 2)

    # 组件2: 稳定性成本（保持不变）
    stability_cost = -(
        w_vel * (vx ** 2 + vy ** 2)
        + w_angle * abs(angle)
        + w_angvel * abs(omega)
    )

    # 组件3: 连续渐进着陆奖励（替代原稀疏 soft_landing_bonus）
    dist = (nx ** 2 + ny ** 2) ** 0.5
    vel_sq = vx ** 2 + vy ** 2

    proximity = 4.0 * (2.718281828 ** (-dist ** 2 / 0.5))
    velocity_ok = 4.0 * (2.718281828 ** (-vel_sq / 0.08))
    angle_ok = 4.0 * (2.718281828 ** (-angle ** 2 / 0.02))
    contact = 4.0 * (left_contact + right_contact) / 2.0

    approach_landing_reward = proximity + velocity_ok + angle_ok + contact

    # 合成总奖励
    total_reward = distance_cost + stability_cost + approach_landing_reward

    # 组件字典
    components = {
        'distance_cost': distance_cost,
        'stability_cost': stability_cost,
        'approach_landing_reward': approach_landing_reward
    }

    return float(total_reward), components