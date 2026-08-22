# 设计理由

**修改了什么组件**：将 `stability_quad_penalty`（全时二次惩罚）替换为 `stability_tilt_hinge_penalty`（仅躯干倾角超安全阈值时的线性惩罚）。

**为什么这样改**：
- **历史证据**：iter 1 使用 `posture_hinge_penalty` 得分 -57.47，iter 2 改成二次惩罚后恶化为 -62.54，说明 hinge 比二次惩罚更有效。二次惩罚在倾角很小时梯度微弱，无法阻止恶化，而一旦倾角变大惩罚急剧上升时 agent 已经接近摔倒，起不到预防作用。
- **信号覆盖审计**：当前奖励仅使用了 hull_angle、hull_angvel、horizontal_speed，缺失垂直速度、接地、关节扭矩等可预警摔倒的信号。在有限修改（每轮只改一个组件）约束下，优先用 hinge 增强 hull_angle 的预警能力，是代价最小的改进。
- **数学形式**：从 `-w * angle^2` 改为 `-w * max(0, |angle| - SAFE_TILT)`。安全阈值 `SAFE_TILT = 0.3` 设在终止边界 0.6 的 50%（略低于 60‑80%，以便与 gate 配合），允许小幅晃动不惩罚，一旦接近危险立即给出线性增长的代价，训练能更早感知“正在变糟”。

**系数校准**：
- 主信号 `gated_forward_speed` 的 per‑step 均值 ≈ 65.35 / 167.4 ≈ 0.39。
- 在临界倾角 0.6 时 hinge 惩罚 = `-TILT_WEIGHT * (0.6 - 0.3)` = `-0.5 * 0.3 = -0.15`，仅为主信号 per‑step 的 38%，且仅在严重倾斜时出现，整体平均惩罚负担远低于 0.5× 主信号，符合设计校准。
- 安全行走区间（|hull_angle| ≤ 0.3）内惩罚为 0，避免了持续压迫探索。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Bipedal locomotion reward for rough terrain:
    - Primary: forward velocity reward with soft health gate based on hull stability
    - Constraint: hinge-style tilt penalty that activates only beyond a safe angle.
    """

    # ==================== Extract signals ====================
    next_hull_angle = next_obs[0]
    horizontal_speed = next_obs[2]

    # ==================== Constants ====================
    # Gate thresholds (unchanged)
    TILT_CRITICAL = 0.6
    TILT_WARNING_START = 0.25
    TILT_WARNING_MARGIN = 0.35

    # Hinge penalty for tilt
    SAFE_TILT = 0.3          # below this no penalty
    TILT_WEIGHT = 0.5        # linear slope above SAFE_TILT

    # Forward weight
    FORWARD_WEIGHT = 2.0

    # ==================== Component A: Forward progress with soft health gate ====================
    abs_tilt = abs(next_hull_angle)
    if abs_tilt <= TILT_WARNING_START:
        gate = 1.0
    elif abs_tilt >= TILT_CRITICAL:
        gate = 0.0
    else:
        gate = (TILT_CRITICAL - abs_tilt) / TILT_WARNING_MARGIN

    forward_reward = FORWARD_WEIGHT * horizontal_speed ** 2
    gated_forward = gate * forward_reward

    # ==================== Component B: Hinge tilt stability penalty ====================
    # Penalize only when tilt exceeds SAFE_TILT, linearly up to CRITICAL.
    tilt_excess = max(0.0, abs_tilt - SAFE_TILT)
    tilt_penalty = TILT_WEIGHT * tilt_excess
    stability_hinge_penalty = -tilt_penalty

    # ==================== Total reward ====================
    total_reward = gated_forward + stability_hinge_penalty

    # ==================== Components dict ====================
    components = {
        'gated_forward_speed': gated_forward,
        'stability_tilt_hinge_penalty': stability_hinge_penalty
    }

    return float(total_reward), components
```

# 诊断摘要
- **audit**: 信号覆盖不全（未使用接地、垂直速度等），但最优先修复方向是将无效的二次惩罚替换为与 iter‑1 一致的 hinge 惩罚，以提供更及时的摔倒预警。
- **behavior**: agent 试图前进但频繁摔倒（50% episode 在 150 步内终止，平均 len ~167），导致累积 forward 奖励不足以抵消负分。
- **signal**: 原 stability_quad_penalty 在全区域施加微小惩罚，在危险区梯度不足，不能有效阻止倾角恶化；hinge 将惩罚集中在危险区间，给出强烈但局部的修正信号。
- **level**: Level 2
- **hypothesis**: Hinge 惩罚在倾角超出安全阈值后立即给出线性增长的代价，使 agent 能更早学会“维持小倾角”，减少摔倒次数，从而提高 episode 生存长度和总 forward 奖励。
- **risk**: 如果 TILT_WEIGHT 过大可能过分压抑必要的大幅度姿态调整，导致前进速度下降；但当前系数经过校准（临界惩罚仅为主信号的 38%），风险可控。