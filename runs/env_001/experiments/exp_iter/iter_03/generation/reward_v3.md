好的，这是根据上一轮奖励函数和迭代上下文生成的修订版本。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observations
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
    
    # 1. Main learning signal: progress_delta_reward (kept)
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_reward = 10.0 * progress_delta
    
    # 2. Distance anchor reward (newly added)
    # Provides a small, continuous incentive to be near the target,
    # which helps when progress signal is noisy or sparse.
    distance_anchor = -0.5 * next_dist
    
    # 3. Smoother landing-quality shaping (kept from previous revision)
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    landing_quality = 0.0
    distance_factor = max(0.0, 1.0 - next_dist / 0.5)
    landing_quality += 1.0 * distance_factor
    speed_factor = max(0.0, 1.0 - speed / 0.5)
    landing_quality += 1.0 * speed_factor
    angle_factor = max(0.0, 1.0 - abs(next_body_angle) / 0.2)
    landing_quality += 1.0 * angle_factor
    both_contact = (next_left_contact > 0.5) and (next_right_contact > 0.5)
    contact_bonus = 0.5 if both_contact else 0.0
    landing_quality += contact_bonus
    landing_shaping_reward = 0.5 * landing_quality
    
    # 4. Stability penalty (weakened to prevent dominance)
    angle_penalty = -0.1 * abs(next_body_angle)
    angular_vel_penalty = -0.05 * abs(next_angular_vel)
    speed_penalty = -0.1 * speed
    stability_penalty = angle_penalty + angular_vel_penalty + speed_penalty
    
    # Total reward
    total_reward = progress_reward + distance_anchor + stability_penalty + landing_shaping_reward
    
    # Components dict
    components = {
        "progress_reward": progress_reward,
        "distance_anchor": distance_anchor,
        "stability_penalty": stability_penalty,
        "landing_shaping_reward": landing_shaping_reward,
        "total_reward": total_reward
    }
    
    return float(total_reward), components
```

### 设计说明

- **保留**：`progress_delta_reward` 作为主要学习信号，`landing_shaping_reward` 作为平滑的着陆引导。
- **削弱**：`stability_penalty` 的权重被显著降低（角度惩罚从 -0.3 降至 -0.1，角速度惩罚从 -0.1 降至 -0.05，速度惩罚从 -0.2 降至 -0.1）。这是为了响应迭代上下文中“稳定性惩罚可能主导进度信号”的诊断，防止其抑制智能体探索移动的行为。
- **修改**：无。
- **新增**：`distance_anchor` 组件。这是一个简单的、与距离成反比的连续奖励（`-0.5 * next_dist`）。它提供了一个稳定的、非零的梯度，鼓励智能体停留在目标附近。这解决了“仅靠进度信号可能较弱”的问题，并为学习提供了一个更平滑的锚点。
- **仍然不使用 terminal_success_reward / terminal_failure_penalty**：因为环境接口中仍然没有提供明确的成功/失败信号。
- **下一轮训练后应该重点观察**：`progress_reward` 的均值是否有所提升，以及 `stability_penalty` 的绝对值是否不再远大于 `progress_reward`。同时，观察 `distance_anchor` 的均值是否稳定为负值，表明智能体正在被拉向目标。