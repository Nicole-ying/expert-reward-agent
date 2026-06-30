# Reward Revision

## 设计说明

### 相比上一轮，保留了什么
- **progress_reward**：核心学习信号，权重保持 10.0，nonzero_rate 接近 1.0，说明有效。
- **approach_reward**：保留，权重 2.0，提供持续接近目标的激励。
- **distance_anchor**：保留，权重 -0.1，作为小幅度距离惩罚锚点。
- **landing_shaping_reward**：保留，但权重进一步降低，因为 landing 在早期训练中可能不是主要目标。

### 削弱或删除了什么
- **stability_penalty**：从上一轮已削弱的基础上再削弱一半（angle_penalty -0.05→-0.025，angular_vel_penalty -0.025→-0.01，speed_penalty -0.01→-0.005），因为其绝对值均值 0.024 与 progress_reward 均值 0.028 相当，可能干扰主信号。
- **landing_shaping_reward**：各子项权重减半（near_target 1.0→0.5，low_speed 0.5→0.25，stable_angle 0.5→0.25，both_contact 0.5→0.25），避免 landing 奖励在早期主导。

### 修改了什么
- **progress_reward**：增加一个基于 training_progress 的线性缩放因子 `1.0 + 0.5 * training_progress`，使 agent 在训练后期更重视进步信号，早期更宽松探索。

### 新增了什么
- **无新增组件**。根据 skeleton 计划，distance_reward 已通过 distance_anchor 实现，无需额外添加。

### 为什么仍然不使用 terminal_success_reward / terminal_failure_penalty
- 环境未提供明确的 success/failure 信号，info 字段不可靠，使用这些奖励会引入虚假信号。

### 下一轮训练后应该重点观察什么
- progress_reward 的均值是否提升（当前 0.028，期望 >0.05）。
- stability_penalty 的绝对值是否降低（当前 0.024，期望 <0.01）。
- mean_eval_reward 是否从 -10.99 提升到正值。
- mean_episode_length 是否稳定在 800+ 或更长。

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
    
    # 1. 主学习信号：progress_delta_reward (保留，增加训练进度缩放)
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_scale = 1.0 + 0.5 * training_progress
    progress_reward = 10.0 * progress_scale * progress_delta
    
    # 2. 稳定约束：stability_penalty (进一步削弱)
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    angle_penalty = -0.025 * abs(next_body_angle)
    angular_vel_penalty = -0.01 * abs(next_angular_vel)
    speed_penalty = -0.005 * speed
    stability_penalty = angle_penalty + angular_vel_penalty + speed_penalty
    
    # 3. 着陆质量塑造：landing_shaping_reward (降低权重)
    near_target = max(0.0, 1.0 - next_dist / 0.5)
    low_speed = max(0.0, 1.0 - speed / 0.3)
    stable_angle = max(0.0, 1.0 - abs(next_body_angle) / 0.2)
    both_contact = 1.0 if (next_left_contact > 0.5) and (next_right_contact > 0.5) else 0.0
    
    landing_shaping_reward = 0.5 * near_target + 0.25 * low_speed + 0.25 * stable_angle + 0.25 * both_contact
    
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