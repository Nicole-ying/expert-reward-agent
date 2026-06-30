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
    angle_penalty = 0.05 * abs(next_body_angle)      # 从0.1降至0.05
    angular_vel_penalty = 0.025 * abs(next_ang_vel)  # 从0.05降至0.025
    speed_penalty = 0.025 * speed                     # 从0.05降至0.025
    stability_penalty = -(angle_penalty + angular_vel_penalty + speed_penalty)
    
    # 3. 平滑接近与着陆引导：smooth_landing_shaping (合并并重写approach_shaping和landing_bonus)
    # 使用连续函数，根据距离、速度和姿态给予平滑奖励，不再依赖硬性接触条件
    near_target_factor = max(0.0, 1.0 - next_dist / 2.0)  # 距离越近，因子越大
    speed_comfort = max(0.0, 1.0 - speed / 1.0)           # 速度越低，奖励越高
    angle_comfort = max(0.0, 1.0 - abs(next_body_angle) / 0.5)  # 姿态越正，奖励越高
    # 当接近目标时，奖励低速和稳定姿态；当远离目标时，此奖励接近于0
    smooth_landing_shaping = 1.5 * near_target_factor * (0.5 * speed_comfort + 0.5 * angle_comfort)
    
    # 4. 距离锚点：distance_reward (保留)
    distance_reward = -0.1 * next_dist
    
    # 组合总奖励
    total_reward = progress_reward + stability_penalty + smooth_landing_shaping + distance_reward
    
    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "smooth_landing_shaping": smooth_landing_shaping,
        "distance_reward": distance_reward,
        "total_reward": total_reward
    }
    
    return float(total_reward), components