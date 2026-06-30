# Response Record

# Reward Revision

## 设计说明

- **保留**：`progress_reward`（核心驱动，证据显示 nonzero_rate 接近 1.0，均值 0.032 合理）
- **保留**：`distance_reward`（作为小锚点，防止 progress-only 在远距离时无梯度）
- **保留**：`stability_penalty`（但进一步削弱，避免主导 progress signal）
- **修改**：`landing_bonus` → 改为更平滑的 `landing_shaping`，不再使用硬阈值触发，而是连续奖励，避免稀疏性导致早期崩溃
- **新增**：`approach_shaping`：当距离较远时，给予额外平滑引导，帮助 agent 在早期获得更稳定的梯度
- **仍然不使用** `terminal_success_reward` / `terminal_failure_penalty`，因为 info 中仍无可靠的成功/失败信号
- **下一轮重点观察**：episode_length 是否增长，external_reward 是否上升，progress_reward 均值是否保持正值

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：progress_delta_reward (保留，核心驱动)
    current_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_reward = 2.0 * progress_delta

    # 稳定约束：stability_penalty (进一步削弱，避免主导progress)
    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    angle_penalty = 0.005 * abs(next_obs[4])  # 从0.01降至0.005
    angular_vel_penalty = 0.002 * abs(next_obs[5])  # 从0.005降至0.002
    speed_penalty = 0.01 * speed  # 从0.02降至0.01
    stability_penalty = -(angle_penalty + angular_vel_penalty + speed_penalty)

    # 平滑接近引导：distance_reward (保留，作为小锚点)
    distance_reward = -0.1 * next_dist

    # 平滑接近引导：approach_shaping (新增，远距离时提供额外梯度)
    approach_shaping = 0.0
    if next_dist > 1.0:
        approach_shaping = -0.2 * next_dist  # 远距离时鼓励靠近

    # 平滑着陆引导：landing_shaping (替代硬阈值landing_bonus)
    near_target = next_dist < 1.0
    low_speed = speed < 1.5
    stable_angle = abs(next_obs[4]) < 0.5
    both_contact = (next_obs[6] > 0.5) and (next_obs[7] > 0.5)
    landing_shaping = 0.0
    if near_target and low_speed and stable_angle:
        landing_shaping = 0.5 * (1.0 - next_dist / 1.0)  # 距离越近奖励越大
        if both_contact:
            landing_shaping += 0.3  # 双脚着地额外奖励

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
