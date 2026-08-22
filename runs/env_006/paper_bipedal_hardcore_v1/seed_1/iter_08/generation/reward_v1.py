def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取观测信号（所有索引均基于 environment_card 声明的 24 维观测）
    hull_angle = obs[0]
    hull_angvel = obs[1]
    horizontal_speed = obs[2]
    vertical_speed = obs[3]

    # ========== 组件 A：前进主奖励 ==========
    # 直接以水平速度作为稠密进展信号，惩罚向后运动
    w_forward = 1.0
    forward_progress = w_forward * horizontal_speed

    # ========== 组件 B：身体倾角稳定性惩罚 ==========
    # 当倾角超过安全阈值（0.5 rad ≈ 28.6°）时施加二次惩罚，越界越多惩罚急剧增大
    angle_threshold = 0.5
    w_angle = 5.0
    angle_error = max(0.0, abs(hull_angle) - angle_threshold)
    stability_angle_penalty = -w_angle * (angle_error ** 2)

    # ========== 组件 C：身体角速度稳定性惩罚 ==========
    # 角速度超过阈值时二次惩罚，抑制急转与即将摔倒的快速旋转
    angvel_threshold = 1.5
    w_angvel = 0.5
    angvel_error = max(0.0, abs(hull_angvel) - angvel_threshold)
    stability_angvel_penalty = -w_angvel * (angvel_error ** 2)

    # ========== 组件 D：垂直速度骤降惩罚 ==========
    # 当机器人向下坠落速度过快时惩罚，预防硬着陆或摔倒
    v_threshold = 0.5
    w_v = 2.0
    v_error = max(0.0, -vertical_speed - v_threshold)  # 只有向下且超过阈值才考虑
    vertical_speed_penalty = -w_v * (v_error ** 2)

    # 总奖励为所有组件的和
    total_reward = forward_progress + stability_angle_penalty + stability_angvel_penalty + vertical_speed_penalty

    components = {
        "forward_progress": forward_progress,
        "stability_angle_penalty": stability_angle_penalty,
        "stability_angvel_penalty": stability_angvel_penalty,
        "vertical_speed_penalty": vertical_speed_penalty
    }

    return float(total_reward), components