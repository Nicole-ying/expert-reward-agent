def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 到原点 (0,0) 的距离
    dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5

    # 主信号：负距离——提供指向原点的密集、单调、方向性梯度
    # dist=2 → -2.0, dist=0 → 0.0，每一步都有明确信号
    # 回退 iter 9 的 progress_delta（均值 0.010，无方向信息）
    distance_reward = -1.0 * dist

    # 辅助信号：连续接近奖励——近距离时提供额外吸引力
    # bounded [0, 2]，dist=0 时=2.0，dist=1 时≈0.33
    proximity_bonus = 2.0 / (1.0 + 5.0 * dist)

    # 稳定约束——保持系数不变，单独归因本轮改动
    vel_penalty = 0.1 * (abs(next_obs[2]) + abs(next_obs[3]))
    angle_penalty = 0.2 * abs(next_obs[4])
    angvel_penalty = 0.05 * abs(next_obs[5])
    stability_penalty = -vel_penalty - angle_penalty - angvel_penalty

    total_reward = distance_reward + proximity_bonus + stability_penalty

    components = {
        'distance_reward': distance_reward,
        'proximity_bonus': proximity_bonus,
        'stability_penalty': stability_penalty,
        'total_reward': total_reward
    }

    return float(total_reward), components