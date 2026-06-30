# Response Record

好的，这是根据上一轮奖励函数和迭代上下文进行的修订。

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
    
    # 2. Stability penalty (weakened: reduced coefficients to prevent dominance)
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    angle_penalty = 0.05 * abs(next_body_angle)  # reduced from 0.1
    angular_vel_penalty = 0.02 * abs(next_angular_vel)  # reduced from 0.05
    speed_penalty = 0.05 * speed  # reduced from 0.1
    stability_penalty = -(angle_penalty + angular_vel_penalty + speed_penalty)
    
    # 3. Simplified landing-quality shaping (kept, unchanged from last iteration)
    near_target_factor = max(0.0, 1.0 - next_distance / 2.0)
    low_speed_factor = max(0.0, 1.0 - speed / 2.0)
    upright_factor = max(0.0, 1.0 - abs(next_body_angle) / 0.5)
    
    landing_quality = near_target_factor * low_speed_factor * upright_factor
    landing_shaping_reward = 1.0 * landing_quality
    
    # 4. Small distance anchor to prevent getting stuck far away (kept, unchanged from last iteration)
    distance_anchor = -0.05 * next_distance
    
    # Combine components
    total_reward = progress_delta_reward + stability_penalty + landing_shaping_reward + distance_anchor
    
    # Build components dict
    components = {
        "progress_delta_reward": progress_delta_reward,
        "stability_penalty": stability_penalty,
        "landing_shaping_reward": landing_shaping_reward,
        "distance_anchor": distance_anchor,
        "total_reward": total_reward
    }
    
    return float(total_reward), components
```

### 设计说明

- **保留**：`progress_delta_reward` 作为主要学习信号，`landing_shaping_reward` 作为着陆引导，`distance_anchor` 作为防止远距离停滞的锚点。这些组件在上一轮中表现稳定，没有证据表明需要修改。
- **削弱**：`stability_penalty` 的系数被整体降低（角度惩罚从0.1降至0.05，角速度惩罚从0.05降至0.02，速度惩罚从0.1降至0.05）。上一轮证据显示该惩罚项的平均值为-0.026，绝对值很小，但为了进一步确保它不会主导 `progress_delta_reward` 信号，我们主动削弱了它。
- **修改**：无。
- **新增**：无。
- **为什么仍然不使用 terminal_success_reward / terminal_failure_penalty**：`info` 中仍然没有可靠的显式成功/失败信号，因此继续推迟使用。
- **下一轮训练后应该重点观察**：`stability_penalty` 的绝对值是否进一步降低，以及 `progress_delta_reward` 是否成为更主导的信号。同时观察 `mean_eval_reward` 和 `mean_episode_length` 的变化，以确认削弱稳定性惩罚是否带来了更好的探索和最终性能。
