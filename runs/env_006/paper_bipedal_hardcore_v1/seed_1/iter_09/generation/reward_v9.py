def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取观测信号
    hull_angle = obs[0]
    horizontal_speed = obs[2]
    leg1_contact = obs[12]
    leg2_contact = obs[13]

    # 组件 A：前进主奖励
    forward_progress = horizontal_speed  # 系数 1.0

    # 组件 B：身体倾角稳定性惩罚（保留）
    angle_threshold = 0.5
    w_angle = 5.0
    angle_error = max(0.0, abs(hull_angle) - angle_threshold)
    stability_angle_penalty = -w_angle * (angle_error ** 2)

    # 组件 C：双脚离地惩罚（新增，替代失效的角速度与垂直速度惩罚）
    w_ground = 0.3
    both_feet_off_ground = (leg1_contact == 0.0 and leg2_contact == 0.0)
    ground_penalty = -w_ground if both_feet_off_ground else 0.0

    total_reward = forward_progress + stability_angle_penalty + ground_penalty

    components = {
        "forward_progress": forward_progress,
        "stability_angle_penalty": stability_angle_penalty,
        "ground_penalty": ground_penalty,
    }
    return float(total_reward), components