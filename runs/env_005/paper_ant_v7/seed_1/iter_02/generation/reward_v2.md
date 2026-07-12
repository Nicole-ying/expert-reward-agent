# 设计理由

## 第 0 步：信号覆盖审计

- **终止 → 前兆**：终止条件为 body_z 越界（<0.2 或 >1.0），当前代码已有 `height_factor` 作为软信号，在边界 0.2 和 1.0 提供线性斜坡。✓ 已覆盖。
- **目标 → 进度**：任务目标为“稳定前进速度”。`forward_gated` 直接用 `vx` 乘 gate，提供前进梯度。✓ 已覆盖。但主信号 per-step 偏弱（~1.05 每步），导致整体得分偏低。
- **效率信号**：动作维度 ≥ 6 → 需要效率约束。已有 `action_penalty`，但 `active_rate=100%`、`episode_sum_mean=-1.45`、`magnitude_share=0.2%`，过于微弱，几乎无约束力。
- **僵尸组件**：`action_penalty` 的 active_rate=100%（非僵尸），但其 magnitude_share=0.2% 极低，形同虚设。此外，缺少对侧向漂移（y 速度）的惩罚——这在运动任务中往往是隐含但关键的质量信号。
- **一句话结论**：当前 reward 漏了侧向漂移惩罚，且 action_penalty 过弱导致能量效率无约束。

## 1. 行为诊断

1. **agent 在做什么**：慢速行走，身体偶尔不稳定（50% terminated，但非早期跌倒）。`forward_gated` 的 per-step 回报约 728.6/694≈1.05，表明平均 vx≈1.3 左右（考虑 gate 打折）。agent 在缓慢前进，但未达到“奔跑”或“快速行走”状态。`action_penalty` 仅 -0.002 每步，对行为无实质影响。

2. **干预哪个目标**：核心干预目标是提升前进速度的同时，引入侧向漂移惩罚以提升运动质量（稳定行走而非摇晃前进）。还需将 action_penalty 调至有效水平，推动节能行为。

3. **这个方向还值得继续吗**：第一轮迭代，骨架设计合理（gate + progress + penalty），方向正确。需要在结构上补全信号，而非重建。

## 2. 选择干预层级

**Level 2 — 结构变换**：增加侧向漂移惩罚组件，修正 action_penalty 系数至有意义水平。

| 证据 | 变换 |
|---|---|
| 缺侧向漂移惩罚（obs[14] 可用但未用） | 新增 `lateral_drift_penalty` 组件 |
| action_penalty magnitude_share=0.2%，过弱 | 系数从 0.0005 提升至 0.005-0.01 |
| upright_factor 使用 `max(0, min(1, body_up_z))` 可能塌缩 | 无界→有界但数学正确。此处 body_up_z ∈ [-1, 1]，`max(0, min(1, body_up_z))` 的结果是：body_up_z ≤ 0 → 0，body_up_z ≥ 1 → 1，中间线性。这是有效的有界因子，不需修改。 |

## 3. 设计校准

- **主信号 per-step** ≈ 728.6/694 ≈ 1.05
- **action_penalty 新系数**：目标 per-step ≤ 0.3 × 1.05 = 0.315。假设平均 action L2 ≈ 5（8 个关节，每个均方约 0.6），则系数应为 0.315/5 ≈ 0.063。保守起见设为 0.01，让 per-step 惩罚 ≈ 0.05，占主信号 5%。
- **lateral_drift_penalty 系数**：目标 per-step ≤ 0.2 × 1.05 = 0.21。假设 v_y 绝对值平均约 0.5 m/s，则系数设为 0.2，per-step ≈ 0.2×0.5²=0.05，占 5%。
- **总惩罚负担**：0.05 + 0.05 = 0.1 << 0.5 × 1.05 = 0.525。安全。

## 4. 代码修改

- **新增组件**：`lateral_drift_penalty = -0.2 * obs[14]**2`，惩罚侧向速度。
- **修正系数**：`w_energy` 从 0.0005 → 0.01，提升至有意义的约束力。
- **保持其他组件不变**。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation signals
    body_z = obs[0]                     # body height
    vx = obs[13]                        # forward velocity (world x)
    vy = obs[14]                        # lateral velocity (world y) - NEW
    quat_x = obs[2]
    quat_y = obs[3]

    # ---- 1. Survival gate: height factor ----
    low_safe  = max(0.0, min(1.0, (body_z - 0.2) / 0.2))
    high_safe = max(0.0, min(1.0, (1.0 - body_z) / 0.2))
    height_factor = min(low_safe, high_safe)

    # ---- 2. Survival gate: uprightness factor ----
    body_up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)
    upright_factor = max(0.0, min(1.0, body_up_z))

    survival_gate = height_factor * upright_factor

    # ---- 3. Main progress component (gated) ----
    w_progress = 1.0
    forward_gated = w_progress * vx * survival_gate

    # ---- 4. Lateral drift penalty (NEW) ----
    w_lateral = 0.2
    lateral_penalty = -w_lateral * (vy ** 2)

    # ---- 5. Energy efficiency term (coefficient corrected) ----
    w_energy = 0.01  # was 0.0005, now 20x stronger
    action_penalty = -w_energy * sum(a**2 for a in action)

    # Total reward
    total_reward = forward_gated + lateral_penalty + action_penalty
    components = {
        'forward_gated': forward_gated,
        'lateral_penalty': lateral_penalty,
        'action_penalty': action_penalty
    }
    return float(total_reward), components
```

---

# 诊断摘要

- **audit**: 当前 reward 漏了侧向漂移惩罚，且 action_penalty 过弱无约束力
- **behavior**: agent 在缓慢行走（vx≈1.3），50% 终止率表明身体控制不稳，有侧向摇晃但未受惩罚
- **signal**: 缺侧向速度惩罚（obs[14] 未使用）；action_penalty = -0.002/step 太弱，对行为无塑造力
- **level**: Level 2
- **hypothesis**: 增加侧向漂移惩罚迫使 agent 学习直线行走（减少无效能量消耗），提升 action_penalty 至 0.01 将鼓励更节能的关节运动，双重信号将提升行走稳定性和速度，因为减少侧向摆动可释放更多动作空间用于前进
- **risk**: 若 agent 当前依赖大幅度侧向摆动来保持平衡，lateral_penalty 初期可能导致更多跌倒（terminated 率上升），但长期应引导学习直线行走策略；action_penalty 提升 20 倍可能过度约束动作探索，若出现停滞可回退至 0.005