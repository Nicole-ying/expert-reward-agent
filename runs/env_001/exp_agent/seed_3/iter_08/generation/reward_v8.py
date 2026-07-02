def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 到原点 (0,0) 的距离
    dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5

    # 主信号1：负距离——提供指向原点的密集梯度
    distance_reward = -1.0 * dist

    # 主信号2：连续接近奖励——替代稀疏的 landing_bonus
    # 数学形态：bounded 在 [0, 2]，dist=0 时=2.0，dist=1 时≈0.33
    # 每一步都有正向信号，越近越强，解决 nonzero_rate=0.19% 的问题
    proximity_bonus = 2.0 / (1.0 + 5.0 * dist)

    # 稳定约束——保持上一轮系数不变，单独归因本轮改动
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