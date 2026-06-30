# Reward Revision – Iteration 6

## 设计说明

### 相比上一轮，保留了什么
- 保留了 `progress_reward` 作为主学习信号，权重不变（10.0）。
- 保留了 `distance_reward` 作为小锚点（-0.1 * next_dist）。
- 保留了 `smooth_landing_shaping` 结构，但调整了权重和阈值。
- 保留了 `contact_landing_bonus` 的 guarded proxy 逻辑。

### 削弱或删除了什么
- **削弱了 `stability_penalty`**：从 `-(0.02*angle + 0.01*ang_vel + 0.01*speed)` 进一步降至 `-(0.01*angle + 0.005*ang_vel + 0.005*speed)`。证据显示其均值仅 -0.005，但可能仍对探索初期造成不必要的抑制。降低后让 progress signal 更主导。
- **删除了 `contact_landing_bonus` 中的 `speed_comfort * angle_comfort` 乘积**：改为固定小奖励 1.0，避免乘积在非理想条件下几乎为零，导致接触奖励几乎不触发。同时将触发阈值从 `next_dist < 0.5` 放宽到 `next_dist < 1.0`，提高触发率。

### 修改了什么
- **修改了 `smooth_landing_shaping`**：权重从 3.0 降至 2.0，因为上一轮中 progress_reward 均值 0.027 已足够，landing shaping 不应过度干扰主信号。同时将 `near_target_factor` 的阈值从 1.5 改回 2.0，使 shaping 在更早阶段开始起作用。

### 新增了什么
- 无新增组件。保持简洁。

### 为什么仍然不使用 terminal_success_reward / terminal_failure_penalty
- 仍然没有显式的 success/failure flag 可用。info 字段不可靠，obs 中无 termination 信号。强行使用会导致训练不稳定或奖励错误。

### 下一轮训练后应该重点观察什么
- `progress_reward` 的均值是否保持在 0.02-0.03 以上，且 nonzero_rate 接近 1.0。
- `stability_penalty` 的绝对值是否进一步下降（理想 < 0.003），避免抑制探索。
- `contact_landing_bonus` 的触发率是否提高（nonzero_rate > 0.1），且均值 > 0.1。
- 外部评估奖励是否继续上升（> 85.68），且 episode length 是否稳定在 800-900 之间。

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
    angle_penalty = 0.01 * abs(next_body_angle)      # 从0.02降至0.01
    angular_vel_penalty = 0.005 * abs(next_ang_vel)   # 从0.01降至0.005
    speed_penalty = 0.005 * speed                     # 从0.01降至0.005
    stability_penalty = -(angle_penalty + angular_vel_penalty + speed_penalty)
    
    # 3. 平滑接近与着陆引导：smooth_landing_shaping (权重降低，阈值放宽)
    near_target_factor = max(0.0, 1.0 - next_dist / 2.0)  # 阈值从1.5改回2.0，更早开始
    speed_comfort = max(0.0, 1.0 - speed / 0.8)           # 保持不变
    angle_comfort = max(0.0, 1.0 - abs(next_body_angle) / 0.3)  # 保持不变
    smooth_landing_shaping = 2.0 * near_target_factor * (0.5 * speed_comfort + 0.5 * angle_comfort)  # 权重从3.0降至2.0
    
    # 4. 接触着陆奖励：contact_landing_bonus (简化，提高触发率)
    contact = (next_left_contact > 0.5) or (next_right_contact > 0.5)
    near_target = next_dist < 1.0  # 阈值从0.5放宽到1.0
    if contact and near_target:
        contact_bonus = 1.0  # 固定小奖励，避免乘积几乎为零
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