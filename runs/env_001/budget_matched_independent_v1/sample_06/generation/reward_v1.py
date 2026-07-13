def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 解包观测变量
    x_pos        = obs[0]
    y_pos        = obs[1]
    x_vel        = obs[2]
    y_vel        = obs[3]
    body_angle   = obs[4]
    ang_vel      = obs[5]
    left_contact = obs[6]
    right_contact = obs[7]

    # 1. 主学习信号：负的欧氏距离，驱动智能体靠近目标垫中心
    distance = (x_pos**2 + y_pos**2)**0.5
    proximity_reward = -1.0 * distance

    # 2. 软着陆速度惩罚：使用距离门控，在接近目标时增强减速要求
    gate = 1.0 / (1.0 + distance)          # 距离越大门越小，避免阻碍早期探索
    speed_sq = x_vel**2 + y_vel**2
    velocity_penalty = -2.0 * speed_sq * gate

    # 3. 稳定性约束：惩罚过大的机体倾斜和旋转速度
    stability_penalty = -0.5 * abs(body_angle) - 0.1 * (ang_vel**2)

    # 4. 双足接触激励：只有双脚都接触垫子时才给予正向奖励
    contact_reward = 2.0 * (left_contact * right_contact)

    # 总奖励
    total_reward = proximity_reward + velocity_penalty + stability_penalty + contact_reward

    # 组件字典，仅包含被累加到 total_reward 中的项
    components = {
        'proximity_reward': proximity_reward,
        'velocity_penalty': velocity_penalty,
        'stability_penalty': stability_penalty,
        'contact_reward': contact_reward
    }

    return total_reward, components