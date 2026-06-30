def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：progress_delta_reward
    # 计算当前位置到目标（0,0）的距离
    current_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_reward = 2.0 * progress_delta  # 权重2.0，鼓励每一步更接近目标
    
    # 稳定约束：stability_penalty
    # 惩罚速度、姿态角和角速度，鼓励稳定接近
    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    angle_penalty = 0.1 * abs(next_obs[4])  # 姿态角惩罚，权重0.1
    angular_vel_penalty = 0.05 * abs(next_obs[5])  # 角速度惩罚，权重0.05
    speed_penalty = 0.2 * speed  # 速度惩罚，权重0.2
    stability_penalty = -(angle_penalty + angular_vel_penalty + speed_penalty)
    
    # 任务完成proxy：soft_landing_proxy
    # 当接近目标、低速、姿态稳定且双支撑接触时给予小奖励
    near_target = next_dist < 0.3
    low_speed = speed < 0.5
    stable_angle = abs(next_obs[4]) < 0.2
    both_contact = (next_obs[6] > 0.5) and (next_obs[7] > 0.5)
    landing_bonus = 1.0 if (near_target and low_speed and stable_angle and both_contact) else 0.0
    
    # 总奖励
    total_reward = progress_reward + stability_penalty + landing_bonus
    
    # 组件记录
    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "landing_bonus": landing_bonus,
        "total_reward": total_reward
    }
    
    return float(total_reward), components