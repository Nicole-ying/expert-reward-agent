# Response Record

# Reward Revision

## 设计说明

### 相比上一轮，保留了什么
- **progress_reward**：保留为主学习信号，权重和计算方式不变（10.0 * progress_delta）。
- **distance_reward**：保留为小锚点（-0.1 * next_dist），防止agent漂移。
- **smooth_landing_shaping**：保留为着陆引导，但进一步调整。
- **contact_landing_bonus**：保留为接触奖励，但调整触发条件。

### 削弱或删除了什么
- **stability_penalty**：大幅削弱。从上一轮的 `-(0.01*|angle| + 0.005*|ang_vel| + 0.005*speed)` 降低到 `-(0.005*|angle| + 0.002*|ang_vel| + 0.002*speed)`。证据显示其均值仅为 -0.002，但可能在高速度时过度惩罚，抑制探索。

### 修改了什么
- **smooth_landing_shaping**：权重从 2.0 降至 1.5，因为上一轮中 progress_reward 已经很强（0.02均值），不需要额外的大幅度 shaping 干扰主信号。
- **contact_landing_bonus**：触发条件从 `next_dist < 1.0` 收紧到 `next_dist < 0.8`，同时要求 `speed < 0.5` 和 `|angle| < 0.2`，确保奖励只在真正稳定着陆时发放，避免随机接触产生噪声。

### 新增了什么
- 无新增组件。当前组件数量（5个）已经足够，证据显示 progress 信号健康，不需要增加复杂度。

### 为什么仍然不使用 terminal_success_reward / terminal_failure_penalty
- 仍然没有显式的 success/failure flag 可用。info 字段不可靠，obs 中没有 termination 信号。强行使用会导致训练不稳定或奖励函数崩溃。

### 下一轮训练后应该重点观察什么
- **progress_reward 的均值**：如果仍然在 0.02 左右，说明主信号健康；如果下降，需要检查是否被其他组件压制。
- **stability_penalty 的绝对值**：应该进一步降低，避免抑制探索。
- **contact_landing_bonus 的触发率**：应该保持低但非零，确保只在真正稳定着陆时发放。
- **外部评估奖励**：如果超过 50，说明改进有效；如果低于 30，需要重新平衡组件。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取观测变量
    x_pos = obs[0]
    y_pos = obs[1]
    x_vel = obs[2]
    y_vel = obs[3]
    body_angle = obs[4]
    ang_vel = obs[5]
    left_contact = obs[6]
    right_contact = obs[7]
    
    next_x_pos = next_obs[0]
    next_y_pos = next_obs[1]
    next_x_vel = next_obs[2]
    next_y_vel = next_obs[3]
    next_body_angle = next_obs[4]
    next_ang_vel = next_obs[5]
    next_left_contact = next_obs[6]
    next_right_contact = next_obs[7]
    
    # 1. 主学习信号：progress_delta_reward (保留，权重不变)
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_reward = 10.0 * progress_delta
    
    # 2. 稳定/安全约束：stability_penalty (进一步削弱权重)
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    angle_penalty = 0.005 * abs(next_body_angle)      # 从0.01降至0.005
    angular_vel_penalty = 0.002 * abs(next_ang_vel)   # 从0.005降至0.002
    speed_penalty = 0.002 * speed                     # 从0.005降至0.002
    stability_penalty = -(angle_penalty + angular_vel_penalty + speed_penalty)
    
    # 3. 平滑接近与着陆引导：smooth_landing_shaping (权重降低)
    near_target_factor = max(0.0, 1.0 - next_dist / 2.0)
    speed_comfort = max(0.0, 1.0 - speed / 0.8)
    angle_comfort = max(0.0, 1.0 - abs(next_body_angle) / 0.3)
    smooth_landing_shaping = 1.5 * near_target_factor * (0.5 * speed_comfort + 0.5 * angle_comfort)  # 权重从2.0降至1.5
    
    # 4. 接触着陆奖励：contact_landing_bonus (收紧触发条件)
    contact = (next_left_contact > 0.5) or (next_right_contact > 0.5)
    near_target = next_dist < 0.8  # 阈值从1.0收紧到0.8
    low_speed = speed < 0.5       # 新增速度条件
    stable_angle = abs(next_body_angle) < 0.2  # 新增角度条件
    if contact and near_target and low_speed and stable_angle:
        contact_bonus = 1.0
    else:
        contact_bonus = 0.0
    
    # 5. 距离锚点：distance_reward (保留)
    distance_reward = -0.1 * next_dist
    
    # 组合总奖励
    total_reward = progress_reward + stability_penalty + smooth_landing_shaping + contact_bonus + distance_reward
    
    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "smooth_landing_shaping": smooth_landing_shaping,
        "contact_landing_bonus": contact_bonus,
        "distance_reward": distance_reward,
        "total_reward": total_reward
    }
    
    return float(total_reward), components
```
