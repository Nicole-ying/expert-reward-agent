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
    
    # 提取接触标志
    left_contact = next_obs[6]
    right_contact = next_obs[7]
    
    # 计算当前位置到目标的距离
    current_distance = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_distance = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    
    # ========== 组件1: 主学习信号 - progress_delta_reward ==========
    # 保留：奖励每一步更接近目标，提供密集的导航引导
    progress_delta = current_distance - next_distance
    progress_delta_reward = 10.0 * progress_delta
    
    # ========== 组件2: 持续引导 - distance_anchor_reward ==========
    # 保留：提供持续的接近目标激励
    distance_anchor_reward = -0.5 * next_distance
    
    # ========== 组件3: 平滑着陆奖励 - smooth_landing_bonus ==========
    # 修改：进一步缩小激活范围（0.4 -> 0.3），降低系数（0.3 -> 0.2）
    # 新增：增加接触标志检查，确保只在有接触时激活
    # 原因：避免过早激活着陆引导干扰主要导航信号，同时确保只在真正着陆时提供引导
    speed = (x_vel ** 2 + y_vel ** 2) ** 0.5
    angle_penalty = abs(body_angle)
    angular_vel_penalty = abs(angular_vel)
    
    # 只在非常接近目标且有接触时激活着陆引导
    if next_distance < 0.3 and (left_contact > 0.5 or right_contact > 0.5):
        # 使用指数衰减形式提供更平滑的引导
        speed_bonus = 1.0 / (1.0 + speed)
        angle_bonus = 1.0 / (1.0 + angle_penalty)
        angular_vel_bonus = 1.0 / (1.0 + angular_vel_penalty)
        smooth_landing_bonus = 0.2 * (speed_bonus + angle_bonus + angular_vel_bonus)
    else:
        smooth_landing_bonus = 0.0
    
    # ========== 总奖励 ==========
    total_reward = progress_delta_reward + distance_anchor_reward + smooth_landing_bonus
    
    # 构建组件字典
    components = {
        "progress_delta_reward": progress_delta_reward,
        "distance_anchor_reward": distance_anchor_reward,
        "smooth_landing_bonus": smooth_landing_bonus,
        "total_reward": total_reward
    }
    
    return float(total_reward), components