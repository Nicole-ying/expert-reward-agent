好的，这是根据上一轮奖励函数和迭代上下文进行的修订版本。

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
    current_distance = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_distance = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_delta = current_distance - next_distance
    progress_delta_reward = 10.0 * progress_delta
    
    # 2. Stability penalty: penalize high velocity, large angle, and high angular velocity (kept, but reduced weight)
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    angle_penalty = 0.3 * abs(next_body_angle)
    angular_vel_penalty = 0.1 * abs(next_angular_vel)
    speed_penalty = 0.2 * speed
    stability_penalty = -(angle_penalty + angular_vel_penalty + speed_penalty)
    
    # 3. Smoother landing-quality shaping (revised from soft_landing_bonus)
    # Provide a continuous reward for being near target, having low speed, and being upright
    # This replaces the hard sparse bonus with a smoother signal
    near_target_factor = max(0.0, 1.0 - next_distance / 2.0)
    low_speed_factor = max(0.0, 1.0 - speed / 2.0)
    upright_factor = max(0.0, 1.0 - abs(next_body_angle) / 0.5)
    
    # Contact bonus is still gated, but now it's a multiplier on a continuous signal
    both_contact = (next_left_contact > 0.5) and (next_right_contact > 0.5)
    contact_factor = 1.0 + 0.5 * both_contact
    
    landing_quality = near_target_factor * low_speed_factor * upright_factor * contact_factor
    landing_shaping_reward = 2.0 * landing_quality
    
    # 4. Small distance anchor to prevent getting stuck far away (kept)
    distance_anchor = -0.1 * next_distance
    
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

- **保留**：`progress_delta_reward` 作为主要学习信号，`distance_anchor` 作为防止远距离停滞的锚点。
- **削弱**：`stability_penalty` 的权重被降低（角度惩罚从0.5降至0.3，角速度惩罚从0.2降至0.1，速度惩罚从0.3降至0.2）。上一轮证据显示该惩罚项主导了奖励信号，导致智能体过早失败。降低其影响可以让智能体更专注于探索前进。
- **修改**：将稀疏的 `soft_landing_bonus` 替换为更平滑的 `landing_shaping_reward`。新奖励基于距离、速度和姿态的连续因子相乘，并加入接触标志作为乘数。这解决了“稀疏完成代理”的问题，为智能体接近目标、减速和保持直立提供了持续的梯度。
- **新增**：无。修订后的 `landing_shaping_reward` 替代了之前的 `soft_landing_bonus`。
- **为什么仍然不使用 terminal_success_reward / terminal_failure_penalty**：`info` 中仍然没有可靠的显式成功/失败信号。使用这些奖励会引入不正确的学习信号。
- **下一轮训练后应该重点观察**：
    1.  `landing_shaping_reward` 的平均值和触发率，确认它是否比之前的 `soft_landing_bonus` 提供了更频繁的指导。
    2.  `stability_penalty` 的平均值，确认其主导地位是否被削弱。
    3.  外部评估奖励和平均回合长度，确认智能体是否能够存活更久并取得更好的分数。