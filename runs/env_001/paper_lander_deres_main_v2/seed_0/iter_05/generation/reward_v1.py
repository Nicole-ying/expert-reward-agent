def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    奖励函数设计：鼓励飞行器平稳接近并双脚着陆目标平台。
    使用联合质量函数 + 接触放大因子，替代进度差奖励。
    """
    # 从 next_obs 提取状态变量
    x_pos = next_obs[0]
    y_pos = next_obs[1]
    x_vel = next_obs[2]
    y_vel = next_obs[3]
    body_angle = next_obs[4]
    angular_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 计算关键量
    distance = (x_pos ** 2 + y_pos ** 2) ** 0.5
    speed = (x_vel ** 2 + y_vel ** 2) ** 0.5
    abs_angle = abs(body_angle)
    abs_angvel = abs(angular_vel)

    # 连续性落地质量函数：同时考量距离、速度、姿态
    alpha_dist = 0.5
    alpha_speed = 0.5
    alpha_angle = 2.0
    alpha_angvel = 1.0

    quality_exp_term = -(
        alpha_dist * distance +
        alpha_speed * speed +
        alpha_angle * abs_angle +
        alpha_angvel * abs_angvel
    )
    quality = 2.718281828 ** quality_exp_term   # exp(negative) ∈ (0, 1]

    # 接触信息：双脚同时接触为1，否则0
    both_contact = left_contact * right_contact   # 乘积：仅当两者都为1时为1

    # 接触放大奖励：在高质量状态下双脚接触获得额外加成
    k_contact = 0.5
    contact_bonus = k_contact * quality * both_contact

    total_reward = quality + contact_bonus

    components = {
        "landing_quality": quality,
        "contact_bonus": contact_bonus
    }

    return float(total_reward), components