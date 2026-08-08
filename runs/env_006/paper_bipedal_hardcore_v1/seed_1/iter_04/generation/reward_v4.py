def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 观测提取
    hull_angle = obs[0]
    hull_angular_velocity = obs[1]
    horizontal_speed = obs[2]
    vertical_speed = obs[3]          # 新增：垂直速度

    # 基础前进奖励
    progress_base = 1.0 * horizontal_speed

    # 稳定性门控：abs(hull_angle) > 0.15 时开始削弱前进奖励
    angle_deviation = abs(hull_angle) - 0.15
    gate = 1.0 - 2.0 * max(0.0, angle_deviation)
    gate = max(0.0, gate)
    progress_reward = progress_base * gate

    # 躯干角速度二次惩罚（保留，轻微抑制剧烈旋转）
    angular_velocity_penalty = -0.1 * (hull_angular_velocity ** 2)

    # 动作效率惩罚
    action_efficiency_penalty = -0.01 * sum(a * a for a in action)

    # 新增：垂直速度惩罚 —— 阻止快速下坠（摔倒前兆）
    # 当 vertical_speed < -0.7 时触发，二次项使惩罚随下坠速度急剧上升
    vertical_velocity_penalty = -0.15 * (max(0.0, -0.7 - vertical_speed) ** 2)

    total_reward = (progress_reward +
                    angular_velocity_penalty +
                    action_efficiency_penalty +
                    vertical_velocity_penalty)
    components = {
        'progress_reward': progress_reward,
        'angular_velocity_penalty': angular_velocity_penalty,
        'action_efficiency_penalty': action_efficiency_penalty,
        'vertical_velocity_penalty': vertical_velocity_penalty
    }
    return float(total_reward), components