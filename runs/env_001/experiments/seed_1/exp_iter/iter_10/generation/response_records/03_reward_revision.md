# Response Record

好的，我们来根据提供的上下文进行奖励函数的修订。

## 设计说明

- **保留**：`progress_delta_reward` 作为主要的学习信号，其结构和系数保持不变。
- **保留**：`distance_reward` 作为一个小型锚点，鼓励智能体向目标移动，系数保持不变。
- **削弱**：`landing_shaping_reward` 的系数从 `1.5` 降低到 `0.5`。上一轮训练的平均奖励（21.92）和最大奖励（179.83）表明，智能体可能已经学会通过快速接近目标来获得高额进度奖励，但着陆质量奖励可能引入了不必要的噪声或局部最优，导致平均奖励偏低。降低其权重可以防止它干扰主要的进度信号。
- **修改**：`landing_shaping_reward` 中的 `near_target_factor` 阈值从 `1.5` 恢复到 `2.0`。上一轮收紧到 `1.5` 可能过于严格，导致智能体在接近目标时才能获得着陆奖励，这不利于早期探索。恢复阈值可以提供一个更平滑的引导。
- **新增**：无。当前组件结构清晰，证据显示没有新增组件的必要。
- **仍然不使用** `terminal_success_reward` 和 `terminal_failure_penalty`，因为 `info` 中仍然没有可靠的显式成功/失败信号。
- **下一轮训练后应重点观察**：`progress_delta_reward` 和 `landing_shaping_reward` 的相对大小，以及平均奖励和最大奖励的变化。如果平均奖励提升但最大奖励下降，说明着陆奖励的削弱可能有效；如果两者都下降，则需要重新评估。

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
    
    # 1. Main learning signal: progress_delta_reward (kept, unchanged)
    current_distance = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_distance = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_delta = current_distance - next_distance
    progress_delta_reward = 10.0 * progress_delta
    
    # 2. Landing-quality shaping (weakened coefficient, relaxed threshold)
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    near_target_factor = max(0.0, 1.0 - next_distance / 2.0)  # relaxed back to 2.0
    low_speed_factor = max(0.0, 1.0 - speed / 2.0)
    upright_factor = max(0.0, 1.0 - abs(next_body_angle) / 0.5)
    
    landing_quality = near_target_factor * low_speed_factor * upright_factor
    landing_shaping_reward = 0.5 * landing_quality  # weakened from 1.5 to 0.5
    
    # 3. Small distance anchor to encourage moving towards target (kept, unchanged)
    distance_reward = -0.1 * next_distance
    
    # Combine components
    total_reward = progress_delta_reward + landing_shaping_reward + distance_reward
    
    # Build components dict
    components = {
        "progress_delta_reward": progress_delta_reward,
        "landing_shaping_reward": landing_shaping_reward,
        "distance_reward": distance_reward,
        "total_reward": total_reward
    }
    
    return float(total_reward), components
```
