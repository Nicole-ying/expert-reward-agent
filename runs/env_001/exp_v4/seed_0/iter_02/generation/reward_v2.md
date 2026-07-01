```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ========== 1. 主学习信号：progress_delta_reward ==========
    # 目标位置为 (0, 0)，因为 obs[0] 和 obs[1] 是相对于目标着陆平台的坐标
    # 当前距离平方
    current_dist_sq = obs[0]**2 + obs[1]**2
    # 下一步距离平方
    next_dist_sq = next_obs[0]**2 + next_obs[1]**2
    
    # 距离减少为正奖励，距离增加为负奖励
    progress_delta = current_dist_sq - next_dist_sq
    # 增大缩放因子，增强学习信号
    progress_scale = 5.0
    progress_delta_reward = progress_delta * progress_scale
    
    # ========== 2. 稳定/安全约束：stability_penalty ==========
    # 惩罚高速、大姿态角和大角速度
    # 使用 next_obs 因为这是动作执行后的状态
    x_vel = next_obs[2]
    y_vel = next_obs[3]
    body_angle = next_obs[4]
    angular_vel = next_obs[5]
    
    # 速度惩罚（大幅降低权重，避免压制进度信号）
    speed = (x_vel**2 + y_vel**2) ** 0.5
    speed_penalty_weight = 0.02  # 从0.1降低到0.02
    speed_penalty = -speed_penalty_weight * speed
    
    # 姿态角惩罚（大幅降低权重）
    angle_penalty_weight = 0.01  # 从0.05降低到0.01
    angle_penalty = -angle_penalty_weight * abs(body_angle)
    
    # 角速度惩罚（大幅降低权重）
    angular_vel_penalty_weight = 0.005  # 从0.02降低到0.005
    angular_vel_penalty = -angular_vel_penalty_weight * abs(angular_vel)
    
    stability_penalty = speed_penalty + angle_penalty + angular_vel_penalty
    
    # ========== 3. 任务完成 proxy：soft_landing_proxy ==========
    # 放宽条件，提高触发率，改为连续塑形
    near_target_threshold = 0.5  # 从0.3放宽到0.5
    low_speed_threshold = 0.5    # 从0.2放宽到0.5
    stable_angle_threshold = 0.3  # 从0.1放宽到0.3
    
    near_target = next_dist_sq < near_target_threshold**2
    low_speed = speed < low_speed_threshold
    stable_angle = abs(body_angle) < stable_angle_threshold
    both_contact = next_obs[6] > 0.5 and next_obs[7] > 0.5
    
    # 改为连续塑形：条件越满足，奖励越大
    soft_landing_bonus_weight = 2.0
    if near_target and low_speed and stable_angle and both_contact:
        soft_landing_proxy = soft_landing_bonus_weight
    elif near_target and low_speed and stable_angle:
        # 接近目标、低速、稳定但未接触：给予部分奖励
        soft_landing_proxy = soft_landing_bonus_weight * 0.5
    elif near_target and low_speed:
        # 接近目标且低速：给予小奖励
        soft_landing_proxy = soft_landing_bonus_weight * 0.2
    else:
        soft_landing_proxy = 0.0
    
    # ========== 4. 动作代价（小权重） ==========
    # 轻微惩罚使用引擎，鼓励燃料效率
    # action 0 是无引擎，其他动作使用引擎
    action_penalty_weight = 0.005  # 从0.01降低到0.005
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
```