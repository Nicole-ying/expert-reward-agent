# Response Record

好的，我们来根据提供的上下文信息，对上一轮的奖励函数进行迭代修订。

## 设计说明

- **保留**：`progress_reward` 作为核心学习信号，其权重和计算方式保持不变。`smooth_landing_shaping` 和 `contact_landing_bonus` 作为着陆引导组件保留，因为它们提供了有效的辅助信号。
- **削弱/删除**：`distance_reward` 被删除。证据显示 `progress_reward` 已经提供了足够的距离导向信号（nonzero_rate 接近 1），额外的负距离奖励可能会与主信号冲突，并可能抑制智能体探索远离目标但能获得更大进步的动作。骨架计划也建议将其作为“小锚点”，但当前证据表明它并非必要。
- **修改**：无。
- **新增**：无。当前组件结构在外部评估中表现良好（mean_eval_reward: 125.44），且没有明确的失败信号指向某个特定组件的问题。因此，本轮修订的重点是简化并移除可能冗余的组件。
- **为什么仍然不使用 terminal_success_reward / terminal_failure_penalty**：`info` 字典中仍然没有可靠的 `success` 或 `failure` 信号，因此这些奖励项仍然被禁止。
- **下一轮训练后应该重点观察**：`progress_reward` 的均值是否保持稳定或增长，以及 `smooth_landing_shaping` 和 `contact_landing_bonus` 的触发率是否合理。同时，需要观察移除 `distance_reward` 后，智能体是否会出现远离目标的行为。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取观测变量
    x_pos = obs[0]
    y_pos = obs[1]
    next_x_pos = next_obs[0]
    next_y_pos = next_obs[1]
    next_x_vel = next_obs[2]
    next_y_vel = next_obs[3]
    next_body_angle = next_obs[4]
    next_left_contact = next_obs[6]
    next_right_contact = next_obs[7]

    # 1. 主学习信号：progress_delta_reward (保留，权重不变)
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_reward = 10.0 * progress_delta

    # 2. 平滑接近与着陆引导：smooth_landing_shaping (保留，权重不变)
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    near_target_factor = max(0.0, 1.0 - next_dist / 2.0)
    speed_comfort = max(0.0, 1.0 - speed / 0.8)
    angle_comfort = max(0.0, 1.0 - abs(next_body_angle) / 0.3)
    smooth_landing_shaping = 1.2 * near_target_factor * (0.5 * speed_comfort + 0.5 * angle_comfort)

    # 3. 接触着陆奖励：contact_landing_bonus (保留)
    contact = (next_left_contact > 0.5) or (next_right_contact > 0.5)
    near_target = next_dist < 0.8
    low_speed = speed < 0.5
    stable_angle = abs(next_body_angle) < 0.2
    if contact and near_target and low_speed and stable_angle:
        contact_bonus = 1.0
    else:
        contact_bonus = 0.0

    # 组合总奖励 (移除了 distance_reward)
    total_reward = progress_reward + smooth_landing_shaping + contact_bonus

    components = {
        "progress_reward": progress_reward,
        "smooth_landing_shaping": smooth_landing_shaping,
        "contact_landing_bonus": contact_bonus,
        "total_reward": total_reward
    }

    return float(total_reward), components
```
