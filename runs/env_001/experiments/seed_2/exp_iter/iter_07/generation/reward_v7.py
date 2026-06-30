def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取观测变量
    x_pos = obs[0]
    y_pos = obs[1]
    x_vel = obs[2]
    y_vel = obs[3]
    body_angle = obs[4]
    ang_vel = obs[5]
    left_contact = obs[6]
    right_contact = obs[7]
    
    next_x_pos = next_obs[0]
    next_y_pos = next_obs[1]
    next_x_vel = next_obs[2]
    next_y_vel = next_obs[3]
    next_body_angle = next_obs[4]
    next_ang_vel = next_obs[5]
    next_left_contact = next_obs[6]
    next_right_contact = next_obs[7]
    
    # 1. 主学习信号：progress_delta_reward (保留，权重不变)
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_reward = 10.0 * progress_delta
    
    # 2. 稳定/安全约束：stability_penalty (进一步削弱权重)
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    angle_penalty = 0.005 * abs(next_body_angle)      # 从0.01降至0.005
    angular_vel_penalty = 0.002 * abs(next_ang_vel)   # 从0.005降至0.002
    speed_penalty = 0.002 * speed                     # 从0.005降至0.002
    stability_penalty = -(angle_penalty + angular_vel_penalty + speed_penalty)
    
    # 3. 平滑接近与着陆引导：smooth_landing_shaping (权重降低)
    near_target_factor = max(0.0, 1.0 - next_dist / 2.0)
    speed_comfort = max(0.0, 1.0 - speed / 0.8)
    angle_comfort = max(0.0, 1.0 - abs(next_body_angle) / 0.3)
    smooth_landing_shaping = 1.5 * near_target_factor * (0.5 * speed_comfort + 0.5 * angle_comfort)  # 权重从2.0降至1.5
    
    # 4. 接触着陆奖励：contact_landing_bonus (收紧触发条件)
    contact = (next_left_contact > 0.5) or (next_right_contact > 0.5)
    near_target = next_dist < 0.8  # 阈值从1.0收紧到0.8
    low_speed = speed < 0.5       # 新增速度条件
    stable_angle = abs(next_body_angle) < 0.2  # 新增角度条件
    if contact and near_target and low_speed and stable_angle:
        contact_bonus = 1.0
    else:
        contact_bonus = 0.0
    
    # 5. 距离锚点：distance_reward (保留)
    distance_reward = -0.1 * next_dist
    
    # 组合总奖励
    total_reward = progress_reward + stability_penalty + smooth_landing_shaping + contact_bonus + distance_reward
    
    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "smooth_landing_shaping": smooth_landing_shaping,
        "contact_landing_bonus": contact_bonus,
        "distance_reward": distance_reward,
        "total_reward": total_reward
    }
    
    return float(total_reward), components