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
    
    # 1. 主学习信号：progress_delta_reward（系数保持2.0，证据显示均值0.032，合理）
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_scale = 2.0
    progress_reward = progress_delta * progress_scale
    
    # 2. 稳定约束：stability_penalty（大幅降低系数，解决dominance问题）
    # 证据显示stability_penalty均值-0.242，是progress_reward的7.5倍，严重主导
    # 将系数从(0.5,0.3,0.2)降低到(0.1,0.05,0.05)
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    angle_penalty = abs(next_body_angle) * 0.1
    angular_vel_penalty = abs(next_ang_vel) * 0.05
    speed_penalty = speed * 0.05
    stability_penalty = -(angle_penalty + angular_vel_penalty + speed_penalty)
    
    # 3. 任务完成proxy：soft_landing_proxy（改为连续shaping，替代稀疏bonus）
    # 证据显示landing_bonus触发率仅0.55%，几乎无效
    # 改为连续shaping：距离越近+速度越低+角度越正，奖励越平滑
    near_target_factor = max(0.0, 1.0 - next_dist / 1.5)  # 距离0时=1，距离1.5时=0
    low_speed_factor = max(0.0, 1.0 - speed / 1.0)       # 速度0时=1，速度1时=0
    stable_angle_factor = max(0.0, 1.0 - abs(next_body_angle) / 0.5)  # 角度0时=1，角度0.5时=0
    landing_shaping = near_target_factor * low_speed_factor * stable_angle_factor * 0.5
    
    # 4. 动作代价：energy_penalty（保持不变，均值-0.013合理）
    engine_use = 1.0 if action != 0 else 0.0
    energy_penalty = -engine_use * 0.1
    
    # 组合总奖励
    total_reward = progress_reward + stability_penalty + landing_shaping + energy_penalty
    
    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "landing_shaping": landing_shaping,
        "energy_penalty": energy_penalty,
        "total_reward": total_reward
    }
    
    return float(total_reward), components
```