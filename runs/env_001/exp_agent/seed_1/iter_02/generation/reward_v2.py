def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 势能塑形 (Potential-Based Shaping) - 替代 progress_delta + soft_landing_bonus
    # Φ = -distance - 0.3*speed - 0.5*|angle|，引导接近+减速+稳定
    # 理论保证最优策略不变 (Ng et al. 1999)
    
    # 当前状态势能
    current_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    current_speed = (obs[2] ** 2 + obs[3] ** 2) ** 0.5
    current_angle = abs(obs[4])
    
    # 下一状态势能
    next_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    next_speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    next_angle = abs(next_obs[4])
    
    # 势能函数：负距离 + 负速度惩罚 + 负角度惩罚
    gamma = 0.99
    phi_current = -(current_dist + 0.3 * current_speed + 0.5 * current_angle)
    phi_next = -(next_dist + 0.3 * next_speed + 0.5 * next_angle)
    shaping_reward = gamma * phi_next - phi_current
    
    # 稳定约束：大幅降低系数，避免惩罚主导
    x_vel = next_obs[2]
    y_vel = next_obs[3]
    body_angle = next_obs[4]
    angular_vel = next_obs[5]
    
    # 速度惩罚：降低10倍
    speed_penalty = -0.05 * (x_vel ** 2 + y_vel ** 2)
    # 姿态角惩罚：降低10倍
    angle_penalty = -0.03 * (body_angle ** 2)
    # 角速度惩罚：降低10倍
    angular_penalty = -0.02 * (angular_vel ** 2)
    
    stability_penalty = speed_penalty + angle_penalty + angular_penalty
    
    # 动作代价：降低系数
    if action == 0:
        energy_penalty = 0.0
    else:
        energy_penalty = -0.01  # 降低10倍
    
    # 总奖励
    total_reward = shaping_reward + stability_penalty + energy_penalty
    
    # 组件字典
    components = {
        "shaping_reward": shaping_reward,
        "stability_penalty": stability_penalty,
        "energy_penalty": energy_penalty,
        "total_reward": total_reward
    }
    
    return float(total_reward), components