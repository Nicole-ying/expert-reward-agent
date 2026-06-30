# Response Record

# Reward Revision

## 设计说明

### 相比上一轮，保留了什么
- **progress_reward**：保留为主学习信号，权重保持10.0，因为其nonzero_rate接近100%，说明持续提供梯度。
- **stability_penalty**：保留但大幅削弱，因为证据显示其均值(-0.045)绝对值已超过progress_reward均值(0.029)，可能主导信号。
- **landing_shaping_reward**：保留加性组合结构，但降低权重以避免过度激励。
- **approach_reward**：保留作为持续接近奖励。
- **distance_anchor**：保留作为小权重锚点。

### 削弱或删除了什么
- **stability_penalty**：所有子项权重减半（angle_penalty从-0.2→-0.1，angular_vel_penalty从-0.1→-0.05，speed_penalty从-0.05→-0.025），使其不再主导progress信号。
- **landing_shaping_reward**：各子项权重从2.0/1.0/1.0/1.0降低到1.0/0.5/0.5/0.5，减少对非目标状态的过度奖励。

### 修改了什么
- 无结构性修改，仅调整权重。

### 新增了什么
- 无新增组件，保持简洁。

### 为什么仍然不使用terminal_success_reward / terminal_failure_penalty
- info中仍无可靠的success/failure信号，使用这些奖励会引入虚假信号，破坏学习稳定性。

### 下一轮训练后应该重点观察什么
- **progress_reward vs stability_penalty 的均值比例**：应确保progress_reward均值至少是stability_penalty绝对值的2倍。
- **mean_eval_reward**：应稳定提升，不再被惩罚项压制。
- **landing_shaping_reward的nonzero_rate**：如果过低，说明着陆条件过于严格，需放宽阈值。

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
    
    # 1. 主学习信号：progress_delta_reward (保留，权重不变)
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_reward = 10.0 * progress_delta
    
    # 2. 稳定约束：stability_penalty (削弱，权重减半)
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    angle_penalty = -0.1 * abs(next_body_angle)          # 原-0.2
    angular_vel_penalty = -0.05 * abs(next_angular_vel)  # 原-0.1
    speed_penalty = -0.025 * speed                       # 原-0.05
    stability_penalty = angle_penalty + angular_vel_penalty + speed_penalty
    
    # 3. 着陆质量塑造：landing_shaping_reward (降低权重)
    near_target = max(0.0, 1.0 - next_dist / 0.5)
    low_speed = max(0.0, 1.0 - speed / 0.3)
    stable_angle = max(0.0, 1.0 - abs(next_body_angle) / 0.2)
    both_contact = 1.0 if (next_left_contact > 0.5) and (next_right_contact > 0.5) else 0.0
    
    landing_shaping_reward = 1.0 * near_target + 0.5 * low_speed + 0.5 * stable_angle + 0.5 * both_contact
    
    # 4. 接近奖励：approach_reward (保留，权重不变)
    approach_reward = 2.0 * max(0.0, 1.0 - next_dist / 1.0)
    
    # 5. 小权重距离锚点：distance_anchor (保留)
    distance_anchor = -0.1 * next_dist
    
    # 组合总奖励
    total_reward = progress_reward + stability_penalty + landing_shaping_reward + approach_reward + distance_anchor
    
    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "landing_shaping_reward": landing_shaping_reward,
        "approach_reward": approach_reward,
        "distance_anchor": distance_anchor,
        "total_reward": total_reward
    }
    
    return float(total_reward), components
```
