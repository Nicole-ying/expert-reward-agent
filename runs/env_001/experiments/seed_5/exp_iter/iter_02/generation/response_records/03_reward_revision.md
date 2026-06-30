# Response Record

好的，这是根据上一轮奖励函数和迭代上下文生成的修订版本。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取观测变量
    x_pos = obs[0]
    y_pos = obs[1]
    x_vel = obs[2]
    y_vel = obs[3]
    body_angle = obs[4]
    angular_vel = obs[5]
    left_contact = obs[6]
    right_contact = obs[7]
    
    next_x_pos = next_obs[0]
    next_y_pos = next_obs[1]
    next_x_vel = next_obs[2]
    next_y_vel = next_obs[3]
    next_body_angle = next_obs[4]
    next_angular_vel = next_obs[5]
    next_left_contact = next_obs[6]
    next_right_contact = next_obs[7]
    
    # 1. 主学习信号：progress_delta_reward (保留)
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_reward = 10.0 * progress_delta
    
    # 2. 稳定约束：stability_penalty (保留，但削弱)
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    angle_penalty = -0.3 * abs(next_body_angle)
    angular_vel_penalty = -0.2 * abs(next_angular_vel)
    speed_penalty = -0.1 * speed
    stability_penalty = angle_penalty + angular_vel_penalty + speed_penalty
    
    # 3. 着陆质量塑造：landing_quality_shaping (修订自 soft_landing_bonus)
    # 使用连续信号替代稀疏奖励，根据接近目标、低速、姿态稳定和双接触的程度给予奖励
    near_target = max(0.0, 1.0 - next_dist / 0.5)
    low_speed = max(0.0, 1.0 - speed / 0.3)
    stable_angle = max(0.0, 1.0 - abs(next_body_angle) / 0.2)
    both_contact = 1.0 if (next_left_contact > 0.5) and (next_right_contact > 0.5) else 0.0
    
    landing_quality = near_target * low_speed * stable_angle * both_contact
    landing_shaping_reward = 5.0 * landing_quality
    
    # 4. 小权重距离锚点：distance_anchor (保留)
    distance_anchor = -0.1 * next_dist
    
    # 组合总奖励
    total_reward = progress_reward + stability_penalty + landing_shaping_reward + distance_anchor
    
    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "landing_shaping_reward": landing_shaping_reward,
        "distance_anchor": distance_anchor,
        "total_reward": total_reward
    }
    
    return float(total_reward), components
```

### 设计说明

- **保留**：`progress_delta_reward` 作为主要学习信号，`distance_anchor` 作为辅助锚点。
- **削弱**：`stability_penalty` 的系数从 `(-0.5, -0.3, -0.2)` 降低到 `(-0.3, -0.2, -0.1)`，以降低其对 progress signal 的主导作用，避免智能体因过度惩罚而不敢移动。
- **修订**：将稀疏的 `soft_landing_bonus` 替换为连续的 `landing_shaping_reward`。新信号通过 `near_target * low_speed * stable_angle * both_contact` 的乘积，在接近目标、低速、姿态稳定且双接触时提供平滑的奖励，解决了原稀疏奖励触发率过低（0.53%）的问题。
- **未新增**：未添加 `terminal_success_reward` 和 `terminal_failure_penalty`，因为环境仍未提供明确的成功/失败信号。
- **下一轮观察重点**：`landing_shaping_reward` 的触发率和均值是否显著提升；`stability_penalty` 是否不再主导总奖励；外部评估奖励是否改善，以及平均回合长度是否增加。
