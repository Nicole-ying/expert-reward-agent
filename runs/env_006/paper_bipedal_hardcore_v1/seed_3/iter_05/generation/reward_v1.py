def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：向前速度
    horizontal_speed = next_obs[2]
    progress = 2.0 * horizontal_speed

    # 稳定/安全约束：姿态角度超出健康范围时软惩罚（hinge）
    hull_angle = next_obs[0]
    max_allowed_angle = 0.3  # 约 17 度
    posture_penalty = -5.0 * max(0.0, abs(hull_angle) - max_allowed_angle)

    # 稳定/安全约束：角速度惩罚，抑制剧烈摇晃
    ang_vel = next_obs[1]
    ang_vel_penalty = -0.05 * (ang_vel ** 2)

    # 效率/动作代价：轻微二次惩罚，抑制过大关节力矩
    action_cost = -0.01 * (action[0]**2 + action[1]**2 + action[2]**2 + action[3]**2)

    total_reward = progress + posture_penalty + ang_vel_penalty + action_cost
    components = {
        'progress_reward': progress,
        'posture_penalty': posture_penalty,
        'ang_vel_penalty': ang_vel_penalty,
        'action_cost': action_cost
    }
    return float(total_reward), components