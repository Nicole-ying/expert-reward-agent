def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取观测变量
    # 当前位置
    x_pos = obs[0]
    y_pos = obs[1]
    # 下一步位置
    next_x_pos = next_obs[0]
    next_y_pos = next_obs[1]
    # 当前速度
    x_vel = obs[2]
    y_vel = obs[3]
    # 姿态
    body_angle = obs[4]
    angular_vel = obs[5]
    # 接触标志
    left_contact = obs[6]
    right_contact = obs[7]
    
    # 下一步速度（用于稳定性惩罚）
    next_x_vel = next_obs[2]
    next_y_vel = next_obs[3]
    next_body_angle = next_obs[4]
    next_angular_vel = next_obs[5]
    
    # 计算到目标(0,0)的距离
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    
    # ========== 组件1: 主学习信号 - progress_delta_reward ==========
    # 奖励每一步更接近目标
    progress_delta = current_dist - next_dist
    progress_scale = 10.0
    progress_reward = progress_delta * progress_scale
    
    # ========== 组件2: 稳定约束 - stability_penalty ==========
    # 惩罚高速、大姿态角和大角速度
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    angle_penalty = abs(next_body_angle)
    angular_vel_penalty = abs(next_angular_vel)
    
    stability_scale = 0.5
    speed_scale = 0.3
    angle_scale = 0.2
    angular_vel_scale = 0.1
    
    stability_penalty = -(
        speed_scale * speed +
        angle_scale * angle_penalty +
        angular_vel_scale * angular_vel_penalty
    ) * stability_scale
    
    # ========== 组件3: 任务完成proxy - soft_landing_proxy ==========
    # 当接近目标、低速、小角度且双接触时给予小奖励
    near_target_threshold = 0.3
    low_speed_threshold = 0.2
    stable_angle_threshold = 0.2
    
    is_near_target = next_dist < near_target_threshold
    is_low_speed = speed < low_speed_threshold
    is_stable_angle = abs(next_body_angle) < stable_angle_threshold
    is_both_contact = (left_contact > 0.5) and (right_contact > 0.5)
    
    soft_landing_bonus = 0.0
    if is_near_target and is_low_speed and is_stable_angle and is_both_contact:
        soft_landing_bonus = 2.0
    
    # ========== 组件4: 动作代价 - energy_penalty (小权重) ==========
    # 使用引擎（动作1,2,3）时给予小惩罚
    engine_actions = [1, 2, 3]
    energy_penalty_scale = 0.1
    energy_penalty = 0.0
    if action in engine_actions:
        energy_penalty = -energy_penalty_scale
    
    # ========== 总奖励 ==========
    total_reward = progress_reward + stability_penalty + soft_landing_bonus + energy_penalty
    
    # 构建组件字典
    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_bonus": soft_landing_bonus,
        "energy_penalty": energy_penalty,
        "total_reward": total_reward
    }
    
    return float(total_reward), components