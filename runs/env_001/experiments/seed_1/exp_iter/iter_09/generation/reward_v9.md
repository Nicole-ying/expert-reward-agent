# Reward Revision for Env_001

## Design Changes

- **保留**: `progress_delta_reward` 作为主要学习信号，系数保持 10.0 不变。
- **保留**: `landing_shaping_reward` 作为着陆质量塑造，系数保持 1.5。
- **保留**: `distance_reward` 作为小距离锚点，系数保持 -0.1。
- **修改**: 将 `landing_shaping_reward` 中的 `near_target_factor` 阈值从 2.0 调整为 1.5，使着陆质量奖励在更接近目标时才开始生效，避免过早激活导致 agent 在远处就尝试减速着陆。
- **新增**: 无新增组件。
- **仍然不使用 terminal_success_reward / terminal_failure_penalty**: 因为 info 中仍然没有可靠的 success/failure 信号。
- **下一轮训练后重点观察**: `landing_shaping_reward` 的平均值和触发频率，以及 `progress_delta_reward` 是否仍然主导训练信号。如果 `landing_shaping_reward` 触发率仍然很低，考虑进一步降低其生效阈值或改用更平滑的 shaping 方式。

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
    
    # 2. Landing-quality shaping (revised: tightened near_target threshold from 2.0 to 1.5)
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    near_target_factor = max(0.0, 1.0 - next_distance / 1.5)  # tightened threshold
    low_speed_factor = max(0.0, 1.0 - speed / 2.0)
    upright_factor = max(0.0, 1.0 - abs(next_body_angle) / 0.5)
    
    landing_quality = near_target_factor * low_speed_factor * upright_factor
    landing_shaping_reward = 1.5 * landing_quality
    
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