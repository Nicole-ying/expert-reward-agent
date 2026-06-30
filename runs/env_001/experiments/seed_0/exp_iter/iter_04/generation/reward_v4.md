好的，我们来根据上一轮的反馈和诊断信息，对奖励函数进行一次有针对性的修订。

## 修订说明

- **保留**：`progress_reward`（核心学习信号）、`distance_anchor`（基线引导）、`stability_penalty`（稳定性惩罚，但已削弱）、`action_penalty`（效率惩罚）。
- **削弱**：`stability_penalty` 的系数从 `0.2`、`0.1`、`0.1` 降低到 `0.1`、`0.05`、`0.05`，以防止其主导进度信号，这与诊断中“避免速度或稳定性惩罚主导主要进度信号”的建议一致。
- **修改**：`landing_shaping` 组件。上一轮的 `landing_shaping` 是一个复杂的乘积项，可能过于稀疏且难以学习。现在将其替换为一个更平滑、基于距离的“接近奖励”（proximity reward），当智能体靠近目标时提供持续的鼓励。这直接回应了“添加更平滑的接近/着陆引导”的行动计划。
- **新增**：无。本次修订专注于优化现有组件，而不是增加新的组件。
- **仍然不使用** `terminal_success_reward` / `terminal_failure_penalty`，因为环境仍然没有提供明确的成功/失败信号。
- **下一轮训练后应重点观察**：`progress_reward` 和 `proximity_reward` 的均值是否提升，以及外部评估奖励（external eval reward）是否变为正值或接近零。同时观察 `stability_penalty` 的绝对值是否仍然过大。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract relevant observations
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
    
    # Component 1: Progress delta reward (main learning signal)
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_reward = progress_delta * 10.0
    
    # Component 2: Distance anchor (small continuous guidance)
    distance_anchor = -0.1 * next_dist
    
    # Component 3: Stability penalty (weakened to avoid dominating)
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    angle_penalty = abs(next_body_angle) * 0.1
    angular_vel_penalty = abs(next_angular_vel) * 0.05
    speed_penalty = speed * 0.05
    stability_penalty = -(angle_penalty + angular_vel_penalty + speed_penalty)
    
    # Component 4: Proximity reward (smooth approach guidance, replaces landing_shaping)
    # Provides a continuous reward that increases as the agent gets closer to the target.
    # This is a simpler and more direct signal than the previous product-based shaping.
    proximity_reward = 1.0 / (1.0 + 5.0 * next_dist)
    
    # Component 5: Small action penalty (efficiency)
    action_penalty = 0.0
    if action != 0:
        action_penalty = -0.05
    
    # Combine components
    total_reward = progress_reward + distance_anchor + stability_penalty + proximity_reward + action_penalty
    
    # Build components dict
    components = {
        "progress_reward": progress_reward,
        "distance_anchor": distance_anchor,
        "stability_penalty": stability_penalty,
        "proximity_reward": proximity_reward,
        "action_penalty": action_penalty,
        "total_reward": total_reward
    }
    
    return float(total_reward), components
```