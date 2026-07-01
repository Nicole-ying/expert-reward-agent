def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ========== 主学习信号：progress_delta_reward ==========
    # 目标位置假设为 (0, 0)，因为 obs[0] 和 obs[1] 是相对于目标着陆平台的坐标
    # 计算当前距离平方和下一时刻距离平方
    current_dist_sq = obs[0] ** 2 + obs[1] ** 2
    next_dist_sq = next_obs[0] ** 2 + next_obs[1] ** 2
    
    # progress_delta: 正数表示更接近目标
    progress_delta = current_dist_sq - next_dist_sq
    
    # 缩放因子，使奖励值在合理范围
    progress_scale = 2.0
    progress_delta_reward = progress_scale * progress_delta
    
    # ========== 稳定/安全约束：stability_penalty ==========
    # 惩罚高速、大姿态角和大角速度
    # 使用 next_obs 因为动作效果体现在下一状态
    vel_x = next_obs[2]
    vel_y = next_obs[3]
    body_angle = next_obs[4]
    angular_vel = next_obs[5]
    
    # 速度惩罚：鼓励减速接近目标
    speed = (vel_x ** 2 + vel_y ** 2) ** 0.5
    speed_penalty_weight = 0.3
    speed_penalty = -speed_penalty_weight * speed
    
    # 姿态角惩罚：鼓励保持水平姿态（角度接近0）
    angle_penalty_weight = 0.2
    angle_penalty = -angle_penalty_weight * abs(body_angle)
    
    # 角速度惩罚：鼓励稳定姿态
    angular_vel_penalty_weight = 0.1
    angular_vel_penalty = -angular_vel_penalty_weight * abs(angular_vel)
    
    stability_penalty = speed_penalty + angle_penalty + angular_vel_penalty
    
    # ========== 任务完成 proxy：soft_landing_proxy ==========
    # 当飞行器接近目标、速度低、姿态稳定且双支撑接触时给予小奖励
    # 条件：距离近、速度低、姿态角小、双接触
    near_target_threshold = 0.5
    low_speed_threshold = 0.3
    stable_angle_threshold = 0.2
    
    is_near_target = current_dist_sq ** 0.5 < near_target_threshold
    is_low_speed = speed < low_speed_threshold
    is_stable_angle = abs(body_angle) < stable_angle_threshold
    is_both_contact = (next_obs[6] > 0.5) and (next_obs[7] > 0.5)
    
    soft_landing_bonus = 0.0
    if is_near_target and is_low_speed and is_stable_angle and is_both_contact:
        soft_landing_bonus = 1.0
    
    soft_landing_proxy = soft_landing_bonus
    
    # ========== 组合总奖励 ==========
    total_reward = progress_delta_reward + stability_penalty + soft_landing_proxy
    
    # ========== 构建 components dict ==========
    components = {
        "progress_delta_reward": progress_delta_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing_proxy,
        "total_reward": total_reward
    }
    
    return float(total_reward), components