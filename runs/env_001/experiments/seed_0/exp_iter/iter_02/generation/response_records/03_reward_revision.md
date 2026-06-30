# Response Record

# Reward Revision

## 设计说明

### 相比上一轮，保留了什么
- **progress_delta_reward**：保留作为主要学习信号，权重不变（10.0）。
- **stability_penalty**：保留但大幅削弱，避免主导 progress signal。
- **action_penalty**：保留，权重不变（-0.05）。

### 削弱或删除了什么
- **soft_landing_bonus**：删除硬阈值条件（near_target + low_speed + stable_angle + both_contact），因为触发率仅0.65%，几乎不提供学习信号。
- **stability_penalty**：系数从 (0.5, 0.3, 0.2) 降低到 (0.2, 0.1, 0.1)，避免惩罚项主导总奖励。

### 修改了什么
- **新增 landing_shaping**：替代原来的稀疏 soft_landing_bonus。这是一个连续、平滑的着陆质量信号，基于距离、速度、角度和接触的综合评分，使用 sigmoid-like 函数在接近目标时逐渐激活，提供渐进式引导。
- **新增 distance_anchor**：一个小幅度的距离奖励锚点（-0.1 * next_dist），确保即使 progress_delta 为负时仍有基础信号，防止 agent 完全失去方向。

### 新增了什么
- **landing_shaping**：连续着陆质量评分，在距离<1.0时逐渐激活，综合速度、角度、接触信息。
- **distance_anchor**：小幅距离惩罚锚点，提供持续的方向引导。

### 为什么仍然不使用 terminal_success_reward / terminal_failure_penalty
- 环境接口未提供明确的 success/failure flag，info 字段不可靠。
- 使用 terminal 奖励需要环境在终止时提供信号，当前无法安全实现。

### 下一轮训练后应该重点观察什么
- **progress_reward 均值**：是否仍保持正值（>0.1）。
- **landing_shaping 均值**：是否从接近0逐渐增长（>0.05 表示 agent 开始学习着陆）。
- **stability_penalty 均值**：是否不再主导总奖励（绝对值 < progress_reward）。
- **外部评估奖励**：是否从 -112 提升到 -80 以上。
- **平均 episode 长度**：是否从 74 增长到 100+。

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
    # Provides a small penalty for being far from target, ensuring baseline signal
    distance_anchor = -0.1 * next_dist
    
    # Component 3: Stability penalty (weakened to avoid dominating)
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    angle_penalty = abs(next_body_angle) * 0.2
    angular_vel_penalty = abs(next_angular_vel) * 0.1
    speed_penalty = speed * 0.1
    stability_penalty = -(angle_penalty + angular_vel_penalty + speed_penalty)
    
    # Component 4: Smooth landing shaping (replaces sparse soft_landing_bonus)
    # Continuous quality score that activates gradually when near target
    # Uses sigmoid-like scaling to provide smooth gradient
    dist_factor = 1.0 / (1.0 + 5.0 * next_dist)  # ~1.0 when dist=0, ~0.17 when dist=1.0
    speed_quality = 1.0 / (1.0 + 2.0 * speed)     # ~1.0 when speed=0, ~0.33 when speed=1.0
    angle_quality = 1.0 / (1.0 + 5.0 * abs(next_body_angle))  # ~1.0 when angle=0
    contact_bonus = 0.0
    if next_left_contact > 0.5 and next_right_contact > 0.5:
        contact_bonus = 0.5
    
    landing_shaping = 2.0 * dist_factor * speed_quality * angle_quality + contact_bonus
    
    # Component 5: Small action penalty (efficiency)
    action_penalty = 0.0
    if action != 0:
        action_penalty = -0.05
    
    # Combine components
    total_reward = progress_reward + distance_anchor + stability_penalty + landing_shaping + action_penalty
    
    # Build components dict
    components = {
        "progress_reward": progress_reward,
        "distance_anchor": distance_anchor,
        "stability_penalty": stability_penalty,
        "landing_shaping": landing_shaping,
        "action_penalty": action_penalty,
        "total_reward": total_reward
    }
    
    return float(total_reward), components
```
