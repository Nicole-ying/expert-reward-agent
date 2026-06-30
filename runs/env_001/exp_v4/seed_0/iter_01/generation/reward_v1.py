def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ========== 1. 主学习信号：progress_delta_reward ==========
    # 目标位置为 (0, 0)，因为 obs[0] 和 obs[1] 是相对于目标着陆平台的坐标
    # 当前距离平方
    current_dist_sq = obs[0]**2 + obs[1]**2
    # 下一步距离平方
    next_dist_sq = next_obs[0]**2 + next_obs[1]**2
    
    # 距离减少为正奖励，距离增加为负奖励
    progress_delta = current_dist_sq - next_dist_sq
    # 缩放因子，使奖励值在合理范围
    progress_scale = 2.0
    progress_delta_reward = progress_delta * progress_scale
    
    # ========== 2. 稳定/安全约束：stability_penalty ==========
    # 惩罚高速、大姿态角和大角速度
    # 使用 next_obs 因为这是动作执行后的状态
    x_vel = next_obs[2]
    y_vel = next_obs[3]
    body_angle = next_obs[4]
    angular_vel = next_obs[5]
    
    # 速度惩罚（鼓励减速）
    speed = (x_vel**2 + y_vel**2) ** 0.5
    speed_penalty_weight = 0.1
    speed_penalty = -speed_penalty_weight * speed
    
    # 姿态角惩罚（鼓励保持水平）
    angle_penalty_weight = 0.05
    angle_penalty = -angle_penalty_weight * abs(body_angle)
    
    # 角速度惩罚（鼓励稳定）
    angular_vel_penalty_weight = 0.02
    angular_vel_penalty = -angular_vel_penalty_weight * abs(angular_vel)
    
    stability_penalty = speed_penalty + angle_penalty + angular_vel_penalty
    
    # ========== 3. 任务完成 proxy：soft_landing_proxy ==========
    # 当接近目标、低速、姿态稳定且双支撑接触时给予小奖励
    near_target_threshold = 0.3  # 距离阈值
    low_speed_threshold = 0.2    # 速度阈值
    stable_angle_threshold = 0.1  # 角度阈值
    
    near_target = next_dist_sq < near_target_threshold**2
    low_speed = speed < low_speed_threshold
    stable_angle = abs(body_angle) < stable_angle_threshold
    both_contact = next_obs[6] > 0.5 and next_obs[7] > 0.5
    
    soft_landing_bonus_weight = 1.0
    if near_target and low_speed and stable_angle and both_contact:
        soft_landing_proxy = soft_landing_bonus_weight
    else:
        soft_landing_proxy = 0.0
    
    # ========== 4. 动作代价（小权重） ==========
    # 轻微惩罚使用引擎，鼓励燃料效率
    # action 0 是无引擎，其他动作使用引擎
    action_penalty_weight = 0.01
    if action == 0:
        action_penalty = 0.0
    else:
        action_penalty = -action_penalty_weight
    
    # ========== 组合总奖励 ==========
    total_reward = progress_delta_reward + stability_penalty + soft_landing_proxy + action_penalty
    
    # ========== 构建 components dict ==========
    components = {
        "progress_delta_reward": progress_delta_reward,
        "stability_penalty": stability_penalty,
        "speed_penalty": speed_penalty,
        "angle_penalty": angle_penalty,
        "angular_vel_penalty": angular_vel_penalty,
        "soft_landing_proxy": soft_landing_proxy,
        "action_penalty": action_penalty,
        "total_reward": total_reward
    }
    
    return float(total_reward), components