好的，这是根据上一轮奖励函数和迭代上下文生成的修订版本。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：progress_delta_reward (保留，核心驱动)
    current_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_reward = 2.0 * progress_delta

    # 稳定约束：stability_penalty (进一步削弱，避免主导progress)
    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    angle_penalty = 0.001 * abs(next_obs[4])  # 从0.002降至0.001
    angular_vel_penalty = 0.0005 * abs(next_obs[5])  # 从0.001降至0.0005
    speed_penalty = 0.003 * speed  # 从0.005降至0.003
    stability_penalty = -(angle_penalty + angular_vel_penalty + speed_penalty)

    # 平滑接近引导：distance_reward (保留，作为小锚点)
    distance_reward = -0.1 * next_dist

    # 平滑接近引导：approach_shaping (保留，远距离时提供额外梯度)
    approach_shaping = 0.0
    if next_dist > 1.0:
        approach_shaping = -0.2 * next_dist

    # 平滑着陆引导：landing_shaping (保留，替代硬阈值landing_bonus)
    near_target = next_dist < 1.0
    low_speed = speed < 1.5
    stable_angle = abs(next_obs[4]) < 0.5
    both_contact = (next_obs[6] > 0.5) and (next_obs[7] > 0.5)
    landing_shaping = 0.0
    if near_target and low_speed and stable_angle:
        landing_shaping = 0.5 * (1.0 - next_dist / 1.0)
        if both_contact:
            landing_shaping += 0.3

    # 总奖励
    total_reward = progress_reward + stability_penalty + distance_reward + approach_shaping + landing_shaping

    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "distance_reward": distance_reward,
        "approach_shaping": approach_shaping,
        "landing_shaping": landing_shaping,
        "total_reward": total_reward
    }

    return float(total_reward), components
```

### 设计说明

- **保留**：`progress_reward` 作为核心驱动信号，`distance_reward` 作为小锚点，`approach_shaping` 和 `landing_shaping` 作为辅助引导。这些组件在上一轮中表现稳定，没有证据表明它们需要被移除。
- **削弱**：`stability_penalty` 的各个子项（`angle_penalty`, `angular_vel_penalty`, `speed_penalty`）的系数被进一步降低。上一轮证据显示 `stability_penalty` 的绝对值均值（0.00122）与 `progress_reward` 的均值（0.005876）相比仍然偏高，可能对早期探索造成不必要的抑制。本轮将其削弱，以进一步确保 `progress_reward` 的主导地位。
- **修改**：无。
- **新增**：无。
- **为什么仍然不使用 terminal_success_reward / terminal_failure_penalty**：`info` 中仍然没有可靠的 `success` 或 `failure` 信号，使用它们会引入不正确的稀疏奖励，破坏学习过程。
- **下一轮训练后应该重点观察**：
    1.  `progress_reward` 的均值是否显著提升，以及其 `nonzero_rate` 是否保持在高位。
    2.  `stability_penalty` 的绝对值均值是否进一步下降，确保其不再主导总奖励。
    3.  外部评估奖励（`mean_eval_reward`）和平均回合长度（`mean_episode_length`）是否继续增长。