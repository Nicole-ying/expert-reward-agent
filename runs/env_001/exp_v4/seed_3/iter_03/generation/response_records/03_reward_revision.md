# Response Record

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
    
    # 1. 主学习信号：progress_delta_reward（恢复到best的2.0系数）
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_scale = 2.0
    progress_reward = progress_delta * progress_scale
    
    # 2. 稳定约束：stability_penalty（恢复到best的系数）
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    angle_penalty = abs(next_body_angle) * 0.5
    angular_vel_penalty = abs(next_ang_vel) * 0.3
    speed_penalty = speed * 0.2
    stability_penalty = -(angle_penalty + angular_vel_penalty + speed_penalty)
    
    # 3. 任务完成proxy：soft_landing_proxy（恢复到best的严格条件，奖励从1.0小幅提升到1.5）
    near_target = next_dist < 0.5
    low_speed = speed < 0.3
    stable_angle = abs(next_body_angle) < 0.2
    both_contact = (next_left_contact > 0.5) and (next_right_contact > 0.5)
    
    landing_bonus = 0.0
    if near_target and low_speed and stable_angle and both_contact:
        landing_bonus = 1.5  # 从1.0小幅提升到1.5，鼓励完成着陆
    
    # 4. 动作代价：energy_penalty（保持不变）
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
