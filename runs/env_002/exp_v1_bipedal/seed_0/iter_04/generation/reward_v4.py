def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ========== 主学习信号：前进速度奖励 ==========
    forward_velocity = next_obs[2]
    fwd_scale = 4.0  # 从2.5提高到4.0，增强前进驱动力，鼓励更快奔跑
    progress_reward = fwd_scale * forward_velocity
    
    # ========== 存活奖励：连续化，鼓励稳定站立 ==========
    hull_angle = abs(next_obs[0])
    hull_angular_vel = abs(next_obs[1])
    
    # 连续存活奖励：角度和角速度都接近0时拿满
    angle_factor = 1.0 - (hull_angle / 0.5) ** 2
    vel_factor = 1.0 - (hull_angular_vel / 2.0) ** 2
    angle_factor = max(0.0, angle_factor)
    vel_factor = max(0.0, vel_factor)
    
    alive_bonus = 0.5 * angle_factor * vel_factor  # 保持不变，存活引导效果良好
    
    # ========== 稳定性约束：适度惩罚 ==========
    angle_penalty_scale = 1.0
    angular_vel_penalty_scale = 0.5
    
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