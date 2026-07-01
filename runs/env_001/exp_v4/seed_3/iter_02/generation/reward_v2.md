```python
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
    
    # 1. 主学习信号：progress_delta_reward（大幅增强）
    # 计算当前位置到目标(0,0)的距离
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_scale = 10.0  # 从2.0提升到10.0，增强学习信号
    progress_reward = progress_delta * progress_scale
    
    # 2. 稳定约束：stability_penalty（大幅降低权重，避免主导）
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    angle_penalty = abs(next_body_angle) * 0.2  # 从0.5降低到0.2
    angular_vel_penalty = abs(next_ang_vel) * 0.1  # 从0.3降低到0.1
    speed_penalty = speed * 0.1  # 从0.2降低到0.1
    stability_penalty = -(angle_penalty + angular_vel_penalty + speed_penalty)
    
    # 3. 任务完成proxy：soft_landing_proxy（提高奖励值，放宽条件）
    near_target = next_dist < 0.8  # 从0.5放宽到0.8
    low_speed = speed < 0.5  # 从0.3放宽到0.5
    stable_angle = abs(next_body_angle) < 0.3  # 从0.2放宽到0.3
    both_contact = (next_left_contact > 0.5) and (next_right_contact > 0.5)
    
    landing_bonus = 0.0
    if near_target and low_speed and stable_angle and both_contact:
        landing_bonus = 2.0  # 从1.0提升到2.0
    
    # 4. 动作代价：energy_penalty（保持小权重）
    engine_use = 1.0 if action != 0 else 0.0
    energy_penalty = -engine_use * 0.1
    
    # 组合总奖励
    total_reward = progress_reward + stability_penalty + landing_bonus + energy_penalty
    
    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "landing_bonus": landing_bonus,
        "energy_penalty": energy_penalty,
        "total_reward": total_reward
    }
    
    return float(total_reward), components
```