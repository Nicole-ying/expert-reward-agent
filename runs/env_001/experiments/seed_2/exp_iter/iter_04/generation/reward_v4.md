好的，我们来根据上一轮的反馈和迭代上下文，对奖励函数进行一次有针对性的修订。

## 修订分析

1.  **保留**：`progress_reward` 作为核心学习信号，其触发率（99.7%）和均值（0.019）表明它正在有效地引导智能体向目标移动。`distance_reward` 作为一个小型锚点，有助于维持向目标靠近的趋势。

2.  **削弱**：`stability_penalty` 的绝对值均值（0.022）与 `progress_reward` 的均值（0.019）相当，这意味着惩罚项正在与主要的学习信号竞争，甚至可能主导梯度。这直接印证了“stability penalty may dominate progress signal”的初步诊断。因此，需要进一步削弱稳定性惩罚的权重，使其成为一个更温和的约束，而不是主要的优化目标。

3.  **修改**：`approach_shaping` 和 `landing_bonus` 在上一轮被新增，旨在提供更平滑的接近引导。然而，从-42.47的外部评估分数来看，效果有限。问题可能在于 `approach_shaping` 的奖励值（最大2.0）仍然不够平滑，且 `landing_bonus` 的硬条件（`both_contact`）触发率极低，导致其几乎没有贡献。因此，将 `approach_shaping` 和 `landing_bonus` 合并为一个更平滑、连续的 `smooth_landing_shaping` 组件。这个新组件将不再依赖硬性的接触条件，而是根据距离、速度和姿态的连续值给予奖励，从而提供更稳定的引导信号。

4.  **新增**：无。本次修订的重点是整合和简化，而不是增加新的组件。

5.  **为什么仍然不使用 terminal_success_reward / terminal_failure_penalty**：`info` 字典中仍然没有可靠的 `success` 或 `failure` 信号。使用这些奖励会导致训练不稳定或奖励信号错误。

6.  **下一轮训练后应该重点观察**：
    - `progress_reward` 的均值是否保持或增长。
    - `stability_penalty` 的绝对值均值是否显著低于 `progress_reward` 的均值。
    - `smooth_landing_shaping` 的均值和触发率，以验证其是否提供了有效的接近引导。
    - 外部评估奖励（`mean_eval_reward`）是否有所提升。

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
    angle_penalty = 0.05 * abs(next_body_angle)      # 从0.1降至0.05
    angular_vel_penalty = 0.025 * abs(next_ang_vel)  # 从0.05降至0.025
    speed_penalty = 0.025 * speed                     # 从0.05降至0.025
    stability_penalty = -(angle_penalty + angular_vel_penalty + speed_penalty)
    
    # 3. 平滑接近与着陆引导：smooth_landing_shaping (合并并重写approach_shaping和landing_bonus)
    # 使用连续函数，根据距离、速度和姿态给予平滑奖励，不再依赖硬性接触条件
    near_target_factor = max(0.0, 1.0 - next_dist / 2.0)  # 距离越近，因子越大
    speed_comfort = max(0.0, 1.0 - speed / 1.0)           # 速度越低，奖励越高
    angle_comfort = max(0.0, 1.0 - abs(next_body_angle) / 0.5)  # 姿态越正，奖励越高
    # 当接近目标时，奖励低速和稳定姿态；当远离目标时，此奖励接近于0
    smooth_landing_shaping = 1.5 * near_target_factor * (0.5 * speed_comfort + 0.5 * angle_comfort)
    
    # 4. 距离锚点：distance_reward (保留)
    distance_reward = -0.1 * next_dist
    
    # 组合总奖励
    total_reward = progress_reward + stability_penalty + smooth_landing_shaping + distance_reward
    
    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "smooth_landing_shaping": smooth_landing_shaping,
        "distance_reward": distance_reward,
        "total_reward": total_reward
    }
    
    return float(total_reward), components
```