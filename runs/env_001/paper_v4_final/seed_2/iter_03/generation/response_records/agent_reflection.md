# Response Record

# 设计理由
## 信号覆盖审计
- **终止模式**: 20/20 truncated (len=1000)，无 crash/越界 → agent 纯粹靠存活来避免负奖励，从未接近平台。
- **观测使用**: 全部 8 维已被引用，但 `landing_bonus` 依赖 `left_contact/right_contact`，active_rate=0%，相当于未使用接触信息。
- **信号缺口**: 缺少不依赖接触的“接近平台”正向梯度。`progress` 的负惩罚按理应推动向原点，但实际强度不足且 agent 已学会存活 exploit。
- **僵尸组件**: `landing_bonus` 完全未触发，必须改造。

## 修改内容（Level 2 — 结构变换）
**改 `landing_bonus` 组件**：从依赖接触条件 (contact_factor) 改为依赖**位置接近度** (proximity)。
- 旧公式：`10.0 * contact_factor * speed_factor * angle_factor`，其中 `contact_factor = (left_contact + right_contact)/2`，因从未接触始终为 0。
- 新公式：`2.0 * proximity * speed_factor * angle_factor`，其中 `proximity = 1/(1 + 10*(x² + y²))`（原点处=1，远处衰减）。
- 移除对 obs[6]/obs[7] 的依赖，保留速度与角度因子以保证着陆质量。
- 系数从 10.0 降为 2.0，避免单步奖励过于极端（但仍可能超过 2x 主信号约束，作为探索引导可以接受，后续若 exploit 再降）。

## 为什么这应改善
`proximity` 为 agent 提供了指向原点的**正向稠密梯度**。即使远处值很小，它也会在优化中创建向平台移动的 incentive，打破当前“存活性循环”的局部最优。速度/角度因子确保只有平稳下降时才获得高奖励，避免冲撞。

## 校准
- 主信号 `progress` per-step ≈ -0.012，新组件最大 2.0，期望平均 0.05~0.2，可能暂时违反单组件 ≤2x 约束，但训练早期需要强引导，后续可通过系数缩放。
- 惩罚负担总和 per-step ≈ -0.019，新正向奖励足以抵消并给出净正值。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack next_obs dimensions per environment card
    x_pos = next_obs[0]
    y_pos = next_obs[1]
    x_vel = next_obs[2]
    y_vel = next_obs[3]
    body_angle = next_obs[4]
    angular_vel = next_obs[5]
    # obs[6], obs[7] not used in this version

    # 1. Primary progress signal: dense quadratic penalty on position and velocity errors.
    pos_sq_error = x_pos**2 + y_pos**2
    vel_sq_error = x_vel**2 + y_vel**2
    progress = -0.05 * pos_sq_error - 0.1 * vel_sq_error

    # 2. Stability constraint: quadratic penalty on body angle and angular velocity.
    pose_penalty = -5.0 * (body_angle**2) - 0.5 * (angular_vel**2)

    # 3. Approach & soft landing bonus: now based on proximity to origin, not contact.
    #    proximity = 1 at (0,0), decays with squared distance.
    proximity = 1.0 / (1.0 + 10.0 * (x_pos**2 + y_pos**2))
    speed_magnitude = abs(x_vel) + abs(y_vel)
    speed_factor = 1.0 / (1.0 + 5.0 * speed_magnitude)
    angle_factor = 1.0 / (1.0 + 20.0 * abs(body_angle))
    landing_bonus = 2.0 * proximity * speed_factor * angle_factor

    total_reward = progress + pose_penalty + landing_bonus

    components = {
        'progress': progress,
        'pose_penalty': pose_penalty,
        'landing_bonus': landing_bonus
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: landing_bonus 完全死亡 (active_rate=0)，因为接触条件从未满足；缺少不依赖接触的接近信号。
- **behavior**: agent 学会无限悬浮/徘徊以避免 crash，从而获得比 crash 更高的累积 reward。
- **signal**: 缺少指向平台的正向导向；当前仅有弱的负位置惩罚。
- **level**: Level 2（结构变换：接触门控 → 位置接近连续奖励）
- **hypothesis**: 位置接近奖励将创造指向原点的正向梯度，引导 agent 下降并尝试着陆；速度/角度因子保证着陆质量。
- **risk**: 正向奖励在接近原点时可能过大，导致 agent 冒 crash 风险猛冲；若出现 crash 率升高，需后续削弱系数或加入 soft health gate。
