def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取位置信息（相对于目标着陆台）
    x_pos = obs[0]
    y_pos = obs[1]
    next_x_pos = next_obs[0]
    next_y_pos = next_obs[1]
    
    # 提取速度信息
    x_vel = next_obs[2]
    y_vel = next_obs[3]
    
    # 提取姿态信息
    body_angle = next_obs[4]
    angular_vel = next_obs[5]
    
    # 提取接触信息
    left_contact = next_obs[6]
    right_contact = next_obs[7]
    
    # 计算当前位置到目标的距离
    current_distance = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_distance = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    
    # ========== 组件1: 主学习信号 - progress_delta_reward ==========
    # 奖励每一步更接近目标，提供密集的导航引导
    progress_delta = current_distance - next_distance
    progress_delta_reward = 10.0 * progress_delta
    
    # ========== 组件2: 稳定约束 - stability_penalty ==========
    # 惩罚高速、大姿态角和大角速度，鼓励稳定接近和着陆
    speed = (x_vel ** 2 + y_vel ** 2) ** 0.5
    angle_penalty = 0.5 * abs(body_angle)
    angular_vel_penalty = 0.2 * abs(angular_vel)
    speed_penalty = 0.3 * speed
    stability_penalty = -(angle_penalty + angular_vel_penalty + speed_penalty)
    
    # ========== 组件3: 任务完成proxy - soft_landing_bonus ==========
    # 当飞行器接近目标、速度低、姿态稳定且双支撑接触时给予小奖励
    near_target = next_distance < 0.5
    low_speed = speed < 0.3
    stable_angle = abs(body_angle) < 0.2
    both_contact = (left_contact > 0.5) and (right_contact > 0.5)
    
    soft_landing_bonus = 0.0
    if near_target and low_speed and stable_angle and both_contact:
        soft_landing_bonus = 2.0
    
    # ========== 总奖励 ==========
    total_reward = progress_delta_reward + stability_penalty + soft_landing_bonus
    
    # 构建组件字典
    components = {
        "progress_delta_reward": progress_delta_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_bonus": soft_landing_bonus,
        "total_reward": total_reward
    }
    
    return float(total_reward), components