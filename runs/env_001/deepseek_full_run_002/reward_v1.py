def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：progress_delta_reward
    # 使用欧几里得距离的负变化作为进度奖励
    current_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_reward = 10.0 * progress_delta  # 权重10，鼓励每一步更接近目标
    
    # 稳定/安全约束1：速度惩罚（轻量）
    # 惩罚过大的线速度，鼓励平稳接近
    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    speed_penalty = -0.5 * speed  # 小权重，避免过度保守
    
    # 稳定/安全约束2：姿态角惩罚（轻量）
    # 惩罚过大的姿态角和角速度，鼓励稳定姿态
    angle_penalty = -0.3 * abs(next_obs[4])  # 姿态角惩罚
    angular_vel_penalty = -0.2 * abs(next_obs[5])  # 角速度惩罚
    stability_penalty = angle_penalty + angular_vel_penalty
    
    # 任务完成proxy：soft_landing_proxy（小权重）
    # 当接近目标、低速、姿态稳定且双支撑接触时给予小奖励
    near_target = next_dist < 0.3  # 距离目标很近
    low_speed = speed < 0.2  # 速度很低
    stable_angle = abs(next_obs[4]) < 0.1  # 姿态接近水平
    both_contact = (next_obs[6] > 0.5) and (next_obs[7] > 0.5)  # 双支撑接触
    
    soft_landing_bonus = 0.0
    if near_target and low_speed and stable_angle and both_contact:
        soft_landing_bonus = 2.0  # 小权重奖励，避免hack
    
    # 总奖励
    total_reward = progress_reward + speed_penalty + stability_penalty + soft_landing_bonus
    
    # 组件记录
    components = {
        "progress_reward": progress_reward,
        "speed_penalty": speed_penalty,
        "stability_penalty": stability_penalty,
        "soft_landing_bonus": soft_landing_bonus,
        "total_reward": total_reward
    }
    
    return float(total_reward), components