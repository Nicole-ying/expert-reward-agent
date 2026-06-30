好的，这是根据上一轮奖励函数和迭代上下文进行的修订。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：progress_delta_reward
    current_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_reward = 2.0 * progress_delta

    # 稳定约束：stability_penalty (削弱)
    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    angle_penalty = 0.05 * abs(next_obs[4])  # 权重从0.1降至0.05
    angular_vel_penalty = 0.025 * abs(next_obs[5])  # 权重从0.05降至0.025
    speed_penalty = 0.1 * speed  # 权重从0.2降至0.1
    stability_penalty = -(angle_penalty + angular_vel_penalty + speed_penalty)

    # 平滑接近引导：distance_reward (新增)
    # 当距离较远时，提供持续的正向引导，避免progress_delta在震荡时信号微弱
    distance_reward = -0.1 * next_dist  # 鼓励靠近目标，权重较小

    # 任务完成proxy：soft_landing_proxy (保留，但条件更宽松)
    near_target = next_dist < 0.5  # 阈值从0.3放宽到0.5
    low_speed = speed < 0.8  # 阈值从0.5放宽到0.8
    stable_angle = abs(next_obs[4]) < 0.3  # 阈值从0.2放宽到0.3
    both_contact = (next_obs[6] > 0.5) and (next_obs[7] > 0.5)
    landing_bonus = 1.0 if (near_target and low_speed and stable_angle and both_contact) else 0.0

    # 总奖励
    total_reward = progress_reward + stability_penalty + distance_reward + landing_bonus

    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "distance_reward": distance_reward,
        "landing_bonus": landing_bonus,
        "total_reward": total_reward
    }

    return float(total_reward), components
```

### 设计说明

- **保留**：`progress_delta_reward` 作为核心学习信号，结构不变。
- **削弱**：`stability_penalty` 的所有子项权重减半。上一轮证据显示其均值 (-0.218) 远大于 progress_reward (0.032)，主导了梯度，导致智能体过早崩溃。削弱后，智能体有更多空间探索移动。
- **修改**：`soft_landing_proxy` 的条件阈值放宽，使其更容易触发，提供更频繁的着陆引导，而不是一个稀疏的硬奖励。
- **新增**：`distance_reward`，一个简单的负距离奖励。当 `progress_delta` 因震荡而微弱时，它能提供一个持续的、平滑的接近目标信号，作为辅助锚点。
- **仍然不使用 terminal_success_reward / terminal_failure_penalty**：因为 `info` 中仍然没有可靠的显式成功/失败信号。
- **下一轮训练后应重点观察**：`progress_reward` 和 `stability_penalty` 的均值是否趋于平衡（例如，progress_reward 的绝对值应大于或等于 stability_penalty 的绝对值），以及 `distance_reward` 的触发率和均值是否稳定。同时关注外部评估奖励是否提升。