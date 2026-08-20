def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主信号：到原点的负距离（目标位置为 (0,0)）
    dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    distance_reward = -1.0 * dist

    # 稳定约束：惩罚速度、倾斜角和角速度的绝对值
    vel_penalty = 0.1 * (abs(next_obs[2]) + abs(next_obs[3]))
    angle_penalty = 0.2 * abs(next_obs[4])
    angvel_penalty = 0.05 * abs(next_obs[5])
    stability_penalty = -vel_penalty - angle_penalty - angvel_penalty

    # 软着陆代理：当满足安全着陆条件时给予一次性正奖励
    landing_bonus = 0.0
    if (dist < 0.1 and 
        abs(next_obs[2]) < 0.2 and abs(next_obs[3]) < 0.2 and 
        abs(next_obs[4]) < 0.05 and abs(next_obs[5]) < 0.1 and 
        next_obs[6] == 1.0 and next_obs[7] == 1.0):
        landing_bonus = 5.0  # 一次性的正奖励

    total_reward = distance_reward + stability_penalty + landing_bonus

    components = {
        'distance_reward': distance_reward,
        'stability_penalty': stability_penalty,
        'landing_bonus': landing_bonus,
        'total_reward': total_reward
    }

    return float(total_reward), components