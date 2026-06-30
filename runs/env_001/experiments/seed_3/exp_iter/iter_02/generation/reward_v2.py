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
    
    # 计算当前位置到目标的距离
    current_distance = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_distance = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    
    # ========== 组件1: 主学习信号 - progress_delta_reward ==========
    # 奖励每一步更接近目标，提供密集的导航引导
    progress_delta = current_distance - next_distance
    progress_delta_reward = 10.0 * progress_delta
    
    # ========== 组件2: 持续引导 - distance_anchor_reward ==========
    # 提供一个微弱的持续信号，鼓励保持接近目标的状态
    distance_anchor_reward = -0.1 * next_distance
    
    # ========== 组件3: 着陆质量塑造 - landing_quality_shaping ==========
    # 根据距离目标远近，动态调整对速度、姿态的约束
    # 当远离目标时，约束很弱；当接近目标时，约束逐渐增强
    speed = (x_vel ** 2 + y_vel ** 2) ** 0.5
    angle_penalty = abs(body_angle)
    angular_vel_penalty = abs(angular_vel)
    speed_penalty = speed
    
    # 使用 sigmoid-like 函数根据距离生成一个 0 到 1 之间的权重
    # 距离越近，权重越高，惩罚越强
    proximity_weight = 1.0 / (1.0 + 10.0 * next_distance)
    
    # 组合惩罚项，并由 proximity_weight 调节
    shaping_penalty = -proximity_weight * (1.0 * angle_penalty + 0.5 * angular_vel_penalty + 0.5 * speed_penalty)
    
    # ========== 总奖励 ==========
    total_reward = progress_delta_reward + distance_anchor_reward + shaping_penalty
    
    # 构建组件字典
    components = {
        "progress_delta_reward": progress_delta_reward,
        "distance_anchor_reward": distance_anchor_reward,
        "landing_quality_shaping": shaping_penalty,
        "total_reward": total_reward
    }
    
    return float(total_reward), components