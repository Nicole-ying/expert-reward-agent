def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ========== 主学习信号：前进速度奖励 ==========
    # 使用水平速度(obs[2])作为前进驱动力
    forward_velocity = next_obs[2]
    fwd_scale = 3.0  # 从2.0提高到3.0，增强前进驱动力
    progress_reward = fwd_scale * forward_velocity
    
    # ========== 存活奖励：连续化，鼓励稳定站立 ==========
    hull_angle = abs(next_obs[0])
    hull_angular_vel = abs(next_obs[1])
    
    # 连续存活奖励：角度和角速度都接近0时拿满，越接近边界越小
    # 使用二次衰减，当角度=0且角速度=0时 reward=0.2
    # 当角度接近0.5或角速度接近2.0时 reward→0
    angle_factor = 1.0 - (hull_angle / 0.5) ** 2
    vel_factor = 1.0 - (hull_angular_vel / 2.0) ** 2
    # 限制在[0,1]范围，避免负值
    angle_factor = max(0.0, angle_factor)
    vel_factor = max(0.0, vel_factor)
    
    alive_bonus = 0.2 * angle_factor * vel_factor  # 连续值，最大0.2
    
    # ========== 稳定性约束：适度惩罚 ==========
    # 惩罚过大的主体角度和角速度
    angle_penalty_scale = 1.0    # 从0.5提高到1.0
    angular_vel_penalty_scale = 0.5  # 从0.3提高到0.5
    
    angle_penalty = angle_penalty_scale * (hull_angle ** 2)
    angular_vel_penalty = angular_vel_penalty_scale * (hull_angular_vel ** 2)
    stability_penalty = -(angle_penalty + angular_vel_penalty)
    
    # ========== 总奖励 ==========
    total_reward = progress_reward + alive_bonus + stability_penalty
    
    # ========== 组件记录 ==========
    components = {
        "progress_reward": progress_reward,
        "alive_bonus": alive_bonus,
        "stability_penalty": stability_penalty
    }
    
    return float(total_reward), components