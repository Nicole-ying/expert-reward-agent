好的，这是根据上一轮奖励函数和迭代上下文进行的修订。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：progress_delta_reward (保留，核心驱动)
    current_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_reward = 2.0 * progress_delta

    # 稳定约束：stability_penalty (进一步削弱，避免主导progress)
    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    angle_penalty = 0.002 * abs(next_obs[4])  # 从0.005降至0.002
    angular_vel_penalty = 0.001 * abs(next_obs[5])  # 从0.002降至0.001
    speed_penalty = 0.005 * speed  # 从0.01降至0.005
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

- **保留了什么**：保留了 `progress_reward` 作为核心驱动信号，保留了 `distance_reward`、`approach_shaping` 和 `landing_shaping` 作为辅助引导。
- **削弱或删除了什么**：根据 `stability_penalty` 的绝对值均值（0.0027）远小于 `progress_reward` 均值（0.0062）的证据，表明稳定性惩罚已经不再主导训练信号。但为了进一步确保它不会在后期成为瓶颈，我再次削弱了 `stability_penalty` 中的各项系数（`angle_penalty` 从0.005降至0.002，`angular_vel_penalty` 从0.002降至0.001，`speed_penalty` 从0.01降至0.005）。
- **修改了什么**：仅修改了 `stability_penalty` 的系数。
- **新增了什么**：没有新增任何组件。
- **为什么仍然不使用 terminal_success_reward / terminal_failure_penalty**：`info` 字典中仍然没有可靠的 `success` 或 `failure` 信号，因此这些奖励项仍然被阻止。
- **下一轮训练后应该重点观察什么**：重点观察 `progress_reward` 的均值是否持续为正且稳定，以及 `stability_penalty` 的绝对值是否仍然远小于 `progress_reward`。同时，观察 `landing_shaping` 的触发率和均值，以评估着陆引导是否有效。