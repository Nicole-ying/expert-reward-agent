# Response Record

# Reward Revision

## 设计说明

### 相比上一轮，保留了什么
- **progress_delta_reward**：保持为主学习信号，权重10.0不变。
- **stability_penalty**：保留，但权重降低以避免主导progress signal。
- **landing_shaping_reward**：保留，但降低权重并简化条件，避免过度惩罚。
- **distance_anchor**：保留，但降低权重，仅作为微弱锚点。

### 削弱或删除了什么
- **stability_penalty**：从 `-(0.3*angle + 0.1*angular_vel + 0.2*speed)` 降低为 `-(0.1*angle + 0.05*angular_vel + 0.1*speed)`。证据显示其绝对值均值仅0.071，但可能在某些episode中过度惩罚，导致agent不敢移动。
- **landing_shaping_reward**：从3.0降低到1.0，并移除contact_factor（因为contact信号不可靠，且可能在不该着陆时触发）。简化后的landing_quality只依赖距离、速度和角度。
- **distance_anchor**：从-0.2降低到-0.05，仅作为微弱锚点防止无限远离。

### 修改了什么
- **landing_shaping_reward**：移除了contact_factor，因为contact信号在非着陆阶段可能产生误导。现在只使用 `near_target_factor * low_speed_factor * upright_factor`，权重从3.0降到1.0。

### 新增了什么
- 无新增组件。证据显示当前主要问题是agent可能过早崩溃或无法有效接近目标，而不是缺少组件。优先让progress_delta_reward主导，同时降低惩罚项。

### 为什么仍然不使用 terminal_success_reward / terminal_failure_penalty
- 仍然没有显式的success/failure信号可用。info中没有可靠字段，obs中也没有terminal flag。强行使用会导致训练不稳定或奖励分布错误。

### 下一轮训练后应该重点观察什么
- **progress_delta_reward的均值**：是否为正，是否主导total_reward。
- **stability_penalty的绝对值均值**：是否显著降低，不再压制progress signal。
- **mean_eval_reward**：是否从-21.82提升，尤其是min_eval_reward是否改善（表明agent不再频繁崩溃）。
- **mean_episode_length**：如果仍为1000，说明agent没有提前终止，但可能原地不动；如果下降，说明agent在尝试移动但失败。

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
    
    # 2. Stability penalty (weakened: reduced weights to avoid dominating progress)
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    angle_penalty = 0.1 * abs(next_body_angle)
    angular_vel_penalty = 0.05 * abs(next_angular_vel)
    speed_penalty = 0.1 * speed
    stability_penalty = -(angle_penalty + angular_vel_penalty + speed_penalty)
    
    # 3. Simplified landing-quality shaping (weakened: removed contact_factor, reduced weight)
    near_target_factor = max(0.0, 1.0 - next_distance / 2.0)
    low_speed_factor = max(0.0, 1.0 - speed / 2.0)
    upright_factor = max(0.0, 1.0 - abs(next_body_angle) / 0.5)
    
    landing_quality = near_target_factor * low_speed_factor * upright_factor
    landing_shaping_reward = 1.0 * landing_quality  # reduced from 3.0 to 1.0
    
    # 4. Small distance anchor to prevent getting stuck far away (weakened)
    distance_anchor = -0.05 * next_distance  # reduced from -0.2 to -0.05
    
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
