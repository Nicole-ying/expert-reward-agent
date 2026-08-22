# Response Record

# 设计理由
删除 `air_stability_penalty` 组件（同时移除 `air_penalty` 和 `vertical_fall_penalty`）。

**问题链**：iter1（仅平衡惩罚+前进奖励）存活长度 246；iter2 加入双脚离地与垂直速度惩罚后长度暴跌至 106；iter3 在保留该惩罚基础上添加地形门控，长度进一步降至 75。`air_stability_penalty` active_rate 仅 13.3%，表明它在大部分 step 不触发，但一旦触发（双脚离地）就会施加较大负奖励（`-0.3` ~ `-1.0×vertical_speed`）。在正常行走的腾空相，双脚短暂离地是必要动作，该惩罚会迫使 agent 抑制任何离地行为，导致步态僵硬、更容易摔倒，最终造成存活步数萎缩。

**变换**：Level 2 结构变换 — 删除一个组件。直接移除 `air_stability_penalty` 的全部计算，回退到“平衡惩罚 + 带地形门控的前进奖励”。这可以消除对自然腾空相的惩罚，让 agent 重新学习更动态的步态，预期存活长度反弹至 150–250 步，score 显著改善。

**系数校准**：保留的 `balance_penalty` 当前 per-step ≈ –0.00078，`forward_reward` per-step ≈ 0.086，惩罚负担比 ≈ 0.009，远低于 0.5x 上限，无需额外调整。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observations
    hull_angle = obs[0]
    hull_angvel = obs[1]
    horizontal_speed = obs[2]
    # vertical_speed no longer needed for air stability penalty

    # ------------------------------------------------------------
    # 1. Terrain awareness gate
    # ------------------------------------------------------------
    lidar_0 = obs[14]
    lidar_1 = obs[15]
    lidar_2 = obs[16]
    lidar_3 = obs[17]
    lidar_4 = obs[18]
    lidar_5 = obs[19]
    lidar_6 = obs[20]
    lidar_7 = obs[21]
    lidar_8 = obs[22]
    lidar_9 = obs[23]

    n_lidar = 10.0
    mean_lidar = (
        lidar_0 + lidar_1 + lidar_2 + lidar_3 + lidar_4 +
        lidar_5 + lidar_6 + lidar_7 + lidar_8 + lidar_9
    ) / n_lidar

    variance = (
        (lidar_0 - mean_lidar) ** 2 +
        (lidar_1 - mean_lidar) ** 2 +
        (lidar_2 - mean_lidar) ** 2 +
        (lidar_3 - mean_lidar) ** 2 +
        (lidar_4 - mean_lidar) ** 2 +
        (lidar_5 - mean_lidar) ** 2 +
        (lidar_6 - mean_lidar) ** 2 +
        (lidar_7 - mean_lidar) ** 2 +
        (lidar_8 - mean_lidar) ** 2 +
        (lidar_9 - mean_lidar) ** 2
    ) / n_lidar

    roughness = variance ** 0.5

    roughness_threshold = 0.3
    roughness_clipped = roughness
    if roughness_clipped > roughness_threshold:
        roughness_clipped = roughness_threshold
    gate = 1.0 - 0.7 * (roughness_clipped / roughness_threshold)  # in [0.3, 1.0]

    forward_reward = horizontal_speed * gate

    # ------------------------------------------------------------
    # 2. Balance penalty
    # ------------------------------------------------------------
    angle_threshold = 0.4
    angvel_threshold = 1.0
    angle_excess = abs(hull_angle) - angle_threshold
    if angle_excess < 0.0:
        angle_excess = 0.0
    angvel_excess = abs(hull_angvel) - angvel_threshold
    if angvel_excess < 0.0:
        angvel_excess = 0.0
    balance_penalty = -3.0 * (angle_excess ** 2) - 0.1 * (angvel_excess ** 2)

    total_reward = forward_reward + balance_penalty

    components = {
        'forward_reward': forward_reward,
        'balance_penalty': balance_penalty,
        'terrain_roughness': roughness,
        'terrain_gate': gate
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 信号缺口在关节状态未用（步态结构），但当前致命伤是 air_stability_penalty 压制自然腾空导致步态崩溃，先清除压制再补步态信号。
- **behavior**: agent 因双脚离地被惩罚而抑制一切腾空动作，步态僵硬，平均 75 步即摔倒。
- **signal**: air_stability_penalty 过强，与步行必须的腾空相冲突。
- **level**: Level 2
- **hypothesis**: 移除双脚离地惩罚后，agent 可恢复自然摆动相，存活步数回升至 150+，整体 score 接近 -10 ~ 0 区间。
- **risk**: 双脚同时离地时间可能略微增加（如跳跃），但平衡惩罚仍会约束极端姿态，后续可引入温和的交替接触鼓励来进一步修复步态。
