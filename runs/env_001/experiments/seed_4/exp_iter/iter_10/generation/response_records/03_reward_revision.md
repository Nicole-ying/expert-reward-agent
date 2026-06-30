# Response Record

# Reward Revision Analysis

## Diagnosis

Based on the iteration context:

1. **progress_reward** is active (nonzero_rate: 0.9995) but its mean is very low (0.006). This suggests the progress signal is present but weak in magnitude.
2. **stability_penalty** is negligible (mean: -0.00026), which is good — it's not dominating.
3. **External evaluation reward** (171.36) and **episode length** (777) suggest the agent is learning to move but not efficiently reaching the target.
4. The skeleton plan suggests keeping progress_delta_reward and considering distance_reward as a small anchor.

## Revision Strategy

- **Keep**: progress_reward as the core driving signal, but increase its coefficient to strengthen the learning gradient.
- **Keep**: stability_penalty at current low levels (not dominating).
- **Keep**: distance_reward as a small anchor.
- **Keep**: approach_shaping for far-distance guidance.
- **Keep**: landing_shaping for near-target behavior.
- **Revise**: Increase progress_reward coefficient from 2.0 to 3.0 to give the agent a stronger gradient toward moving closer to the target.
- **Revise**: Slightly increase distance_reward from -0.1 to -0.15 to provide a bit more pull toward the target.
- **No new components**: The current structure is clean and evidence shows no missing critical signals.

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：progress_delta_reward (保留，系数从2.0提升至3.0以增强梯度)
    current_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_reward = 3.0 * progress_delta

    # 稳定约束：stability_penalty (保留，保持低权重避免主导progress)
    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    angle_penalty = 0.0005 * abs(next_obs[4])
    angular_vel_penalty = 0.0002 * abs(next_obs[5])
    speed_penalty = 0.001 * speed
    stability_penalty = -(angle_penalty + angular_vel_penalty + speed_penalty)

    # 平滑接近引导：distance_reward (保留，系数从-0.1微调至-0.15)
    distance_reward = -0.15 * next_dist

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

## Design Notes

- **保留**：progress_reward（核心驱动）、stability_penalty（低权重约束）、distance_reward（小锚点）、approach_shaping（远距离引导）、landing_shaping（着陆引导）。
- **修改**：progress_reward 系数从 2.0 提升至 3.0，distance_reward 系数从 -0.1 微调至 -0.15。
- **未新增**：当前组件结构已覆盖主要信号，证据显示没有缺失关键信号。
- **仍不使用 terminal_success_reward / terminal_failure_penalty**：info 中仍无可靠的 success/failure 信号。
- **下一轮重点观察**：progress_reward 的 mean 值是否提升（目标 >0.01），以及 external evaluation reward 是否增长。如果 progress_reward 仍低，可考虑进一步增加系数或添加基于速度的辅助引导。
