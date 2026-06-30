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
    angle_penalty = -0.2 * abs(next_body_angle)
    angular_vel_penalty = -0.1 * abs(next_angular_vel)
    speed_penalty = -0.05 * speed
    stability_penalty = angle_penalty + angular_vel_penalty + speed_penalty
    
    # 3. 着陆质量塑造：landing_quality_shaping (修订)
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

- **保留**：`progress_reward` 作为主要学习信号，`landing_shaping_reward` 作为着陆引导，`distance_anchor` 作为小权重距离锚点。
- **削弱**：`stability_penalty` 的各个子项（角度惩罚、角速度惩罚、速度惩罚）的系数均被降低。上一轮证据显示 `stability_penalty` 的绝对值均值（0.126）与 `progress_reward` 均值（0.161）相比过大，可能主导了梯度信号，导致智能体过早失败。削弱后，主学习信号能更有效地引导智能体。
- **修改**：无。
- **新增**：无。
- **为什么仍然不使用 terminal_success_reward / terminal_failure_penalty**：`info` 中仍然没有可靠的显式成功/失败信号，使用它们会违反环境约束。
- **下一轮训练后应该重点观察**：`progress_reward` 和 `stability_penalty` 的均值比例是否更合理（例如，`progress_reward` 的绝对值应显著大于 `stability_penalty`），以及 `mean_episode_length` 是否有所增加，以验证削弱稳定性惩罚是否缓解了过早失败的问题。
