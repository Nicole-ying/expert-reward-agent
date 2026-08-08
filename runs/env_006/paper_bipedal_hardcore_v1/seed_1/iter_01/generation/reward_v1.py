def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主进展：水平速度正向奖励
    horizontal_speed = obs[2]
    progress_reward = 1.0 * horizontal_speed

    # 稳定约束：躯干倾斜角 hinge 惩罚（超出阈值才惩罚）
    hull_angle = obs[0]
    angle_threshold = 0.3  # 容许小幅倾斜，靠近摔倒前开始惩罚
    hinge = max(0.0, abs(hull_angle) - angle_threshold)
    stability_penalty = -0.5 * hinge

    # 效率代价：动作二次惩罚（轻量）
    action_efficiency_penalty = -0.01 * sum(a * a for a in action)

    total_reward = progress_reward + stability_penalty + action_efficiency_penalty
    components = {
        'progress_reward': progress_reward,
        'stability_penalty': stability_penalty,
        'action_efficiency_penalty': action_efficiency_penalty
    }
    return float(total_reward), components