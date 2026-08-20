def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 当前步距离和下一步距离
    old_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    new_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5

    # 主信号1：progress_delta——替代 distance_reward
    # 靠近原点 → 正奖励，远离 → 负惩罚，提供密集的方向性梯度
    # 系数 5.0：远距离时主导（d=2→1.5，delta=0.5，reward=2.5）
    progress_delta_reward = 5.0 * (old_dist - new_dist)

    # 主信号2：proximity_bonus——保持不变
    # 近距离时提供强吸引力（d=0 时=2.0），远距离时弱但始终为正
    proximity_bonus = 2.0 / (1.0 + 5.0 * new_dist)

    # 稳定约束——保持上一轮系数不变
    vel_penalty = 0.1 * (abs(next_obs[2]) + abs(next_obs[3]))
    angle_penalty = 0.2 * abs(next_obs[4])
    angvel_penalty = 0.05 * abs(next_obs[5])
    stability_penalty = -vel_penalty - angle_penalty - angvel_penalty

    total_reward = progress_delta_reward + proximity_bonus + stability_penalty

    components = {
        'progress_delta_reward': progress_delta_reward,
        'proximity_bonus': proximity_bonus,
        'stability_penalty': stability_penalty,
        'total_reward': total_reward
    }

    return float(total_reward), components