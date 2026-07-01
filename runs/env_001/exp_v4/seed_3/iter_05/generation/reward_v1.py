def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：progress_delta_reward
    # 使用欧几里得距离的负变化量作为势能塑形
    current_pos = obs[0:2]  # [x_position, y_position]
    next_pos = next_obs[0:2]  # [x_position, y_position]
    
    # 计算当前步和下一步到目标(0,0)的距离
    current_dist = (current_pos[0]**2 + current_pos[1]**2) ** 0.5
    next_dist = (next_pos[0]**2 + next_pos[1]**2) ** 0.5
    
    # progress_delta: 距离减少为正奖励，增加为负奖励
    progress_delta = current_dist - next_dist
    progress_scale = 2.0
    progress_reward = progress_delta * progress_scale
    
    # 稳定/安全约束：stability_penalty
    # 惩罚高速、大姿态角和大角速度，鼓励稳定接近
    vel_x = next_obs[2]
    vel_y = next_obs[3]
    body_angle = next_obs[4]
    angular_vel = next_obs[5]
    
    # 速度惩罚：平方形式，对高速更敏感
    speed = (vel_x**2 + vel_y**2) ** 0.5
    speed_penalty_weight = 0.1
    speed_penalty = -speed_penalty_weight * speed**2
    
    # 姿态角惩罚：角度偏离0度
    angle_penalty_weight = 0.05
    angle_penalty = -angle_penalty_weight * body_angle**2
    
    # 角速度惩罚
    angular_vel_penalty_weight = 0.02
    angular_vel_penalty = -angular_vel_penalty_weight * angular_vel**2
    
    stability_penalty = speed_penalty + angle_penalty + angular_vel_penalty
    
    # 任务完成 proxy：soft_landing_proxy
    # 当接近目标、低速、小姿态角且双支撑接触时给予小奖励
    near_target_threshold = 0.3
    low_speed_threshold = 0.5
    stable_angle_threshold = 0.2
    
    left_contact = next_obs[6]
    right_contact = next_obs[7]
    
    is_near_target = next_dist < near_target_threshold
    is_low_speed = speed < low_speed_threshold
    is_stable_angle = abs(body_angle) < stable_angle_threshold
    is_both_contact = (left_contact > 0.5) and (right_contact > 0.5)
    
    landing_bonus_weight = 1.0
    if is_near_target and is_low_speed and is_stable_angle and is_both_contact:
        landing_bonus = landing_bonus_weight
    else:
        landing_bonus = 0.0
    
    # 总奖励
    total_reward = progress_reward + stability_penalty + landing_bonus
    
    # 组件字典
    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "landing_bonus": landing_bonus,
        "total_reward": total_reward
    }
    
    return float(total_reward), components