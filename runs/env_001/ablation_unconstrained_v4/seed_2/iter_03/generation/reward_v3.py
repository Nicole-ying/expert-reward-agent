def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 当前与下一状态
    x_old, y_old = obs[0], obs[1]
    x_new, y_new = next_obs[0], next_obs[1]
    vx_new, vy_new = next_obs[2], next_obs[3]
    angle_new = next_obs[4]
    angvel_new = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 到原点的距离（目标平台位于 (0,0)）
    dist_old = (x_old ** 2 + y_old ** 2) ** 0.5
    dist_new = (x_new ** 2 + y_new ** 2) ** 0.5

    # 权重
    w_progress = 5.0
    w_y_progress = 2.0
    w_alignment = 0.5
    w_horizontal_penalty = 0.05
    w_vel = 0.1
    w_att = 0.1
    w_contact = 3.0
    w_ground = 1.0

    # 阈值
    vx_max = 0.5
    vy_max = 0.5
    angle_max = 0.2
    angvel_max = 0.2
    ideal_angle_tol = 0.1
    ideal_vy_tol = 0.2

    # 1. 向目标移动的稠密进展奖励（距离缩短）
    progress = w_progress * (dist_old - dist_new)

    # 2. 垂直下降奖励（直接奖励高度减小）
    y_progress = w_y_progress * (y_old - y_new)

    # 3. 水平对准奖励（接近 x=0 时高奖励）
    alignment = w_alignment * (1.0 / (1.0 + x_new ** 2))

    # 4. 微小水平偏离惩罚（仅在需要时提供软边界）
    horizontal_penalty = -w_horizontal_penalty * (x_new ** 2)

    # 5. 安全速度约束
    excess_vx = max(0.0, abs(vx_new) - vx_max)
    excess_vy = max(0.0, abs(vy_new) - vy_max)
    vel_penalty = -w_vel * (excess_vx ** 2 + excess_vy ** 2)

    # 6. 姿态安全约束
    excess_angle = max(0.0, abs(angle_new) - angle_max)
    excess_angvel = max(0.0, abs(angvel_new) - angvel_max)
    att_penalty = -w_att * (excess_angle ** 2 + excess_angvel ** 2)

    # 7. 接触奖励（任意支撑腿接触 + 质量因子）
    any_contact = max(left_contact, right_contact)
    angle_quality = max(0.0, 1.0 - abs(angle_new) / ideal_angle_tol)
    vy_quality = max(0.0, 1.0 - abs(vy_new) / ideal_vy_tol)
    contact_reward = w_contact * any_contact * angle_quality * vy_quality

    # 8. 接近地面奖励（当高度低于阈值时线性激励）
    ground_threshold = 0.5
    height_near_reward = w_ground * max(0.0, ground_threshold - y_new)

    # 总和
    total_reward = (progress + y_progress + alignment + horizontal_penalty +
                    vel_penalty + att_penalty + contact_reward + height_near_reward)

    components = {
        'progress': progress,
        'y_progress': y_progress,
        'alignment': alignment,
        'horizontal_penalty': horizontal_penalty,
        'vel_penalty': vel_penalty,
        'att_penalty': att_penalty,
        'contact_reward': contact_reward,
        'height_near_reward': height_near_reward
    }

    return float(total_reward), components