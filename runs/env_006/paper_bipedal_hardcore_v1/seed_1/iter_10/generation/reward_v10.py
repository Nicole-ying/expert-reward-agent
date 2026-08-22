def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取观测信号
    hull_angle = obs[0]
    horizontal_speed = obs[2]

    # 组件 A：前进主奖励
    forward_progress = horizontal_speed  # 系数 1.0

    # 组件 B：身体倾角稳定性惩罚
    angle_threshold = 0.5
    w_angle = 5.0
    angle_error = max(0.0, abs(hull_angle) - angle_threshold)
    stability_angle_penalty = -w_angle * (angle_error ** 2)

    # 组件 C：动作效率惩罚（替代失效的 ground_penalty）
    w_action = 0.02
    action_sum_sq = action[0]**2 + action[1]**2 + action[2]**2 + action[3]**2
    action_penalty = -w_action * action_sum_sq

    total_reward = forward_progress + stability_angle_penalty + action_penalty

    components = {
        "forward_progress": forward_progress,
        "stability_angle_penalty": stability_angle_penalty,
        "action_penalty": action_penalty,
    }
    return float(total_reward), components