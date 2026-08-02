# 设计理由

## 本轮改动
将 `soft_landing_penalty`（二次惩罚项，signed_share 82%，active_rate 100%）替换为 `landing_gate`（软着陆门控因子），乘到 `progress` 上。`contact_success_reward` 保持不变（虽几乎不触发，但不产生负奖励，本轮暂不对其动刀）。

## 为什么这样改

- **审计结论**：当前 `soft_landing_penalty` 是造成 episode 过早终止的元凶。它每步产生巨大负奖励（ep_sum_mean 52.4，而 progress 仅 4.14），导致 agent 学会“快速坠毁以终止负奖励”，len 从 1000 暴跌到 501，terminated 率 100%。`contact_success_reward` active_rate 仅 2.6%，几乎永远无法补偿惩罚。
- **行为**：agent 在惩罚压力下选择最快终止路径（crash 或飞出视野），完全放弃生存和着陆尝试。
- **信号缺失**：不是缺少观测，而是惩罚过于强大，淹没了唯一的正信号 `progress`。
- **干预层级**：Level 2 结构变换 —— 独立约束惩罚 → 软件健康门控。消除常驻负奖励，把姿态/速度约束变为 `progress` 的乘性门，唯有保持安全姿态，才能有效获得前进奖励。

## 数学形式与系数校准

**landing_gate**：由四个高斯因子乘积构成，sigma 适当放宽，保证在“不理想但安全”的区域 gate ≥ 0.3。
- `safe_vy = exp(- y_vel² / 0.5²)`
- `safe_vx = exp(- x_vel² / 0.4²)`
- `safe_angle = exp(- angle² / 0.2²)`
- `safe_angvel = exp(- ang_vel² / 0.4²)`
- `landing_gate = safe_vy * safe_vx * safe_angle * safe_angvel`

**总奖励**：`total_reward = landing_gate * progress + contact_success_reward`
- `progress = 10.0 * (dist_curr - dist_next)`（沿用，每步量级 ~0.008）
- `contact_success_reward` 保持原样（200 * contact_flag * 严格高斯因子），但不再被门控乘，以免双重削弱，同时避免它在未接触时产生任何惩罚。

**设计校准**：
- 惩罚负担降到 0，不再超标。
- 门控因子乘到 progress 上，不会产生负值，仅减弱无效步的奖励，驱动 agent 改善姿态而非求死。
- 若 gate 平均约为 0.3~0.4，则 effective progress 每步 ~0.0025~0.003，偏小但足以形成梯度；配合 episode 生存时间延长，累积 progress 将增大，逐渐引导 agent 靠近目标。

## 预期效果
- len 将回升并趋于存活，不再出现主动坠毁。
- 姿态与速度逐步改善，gate 增大，有效 progress 增加，总得分由负转正。
- `contact_success_reward` 仍然稀疏，下一轮可将其替换为更密集的着陆引导信号，进一步加快收敛。

## 风险
- 由于 progress 的原始 per‑step 量级较小，乘 gate 后可能学习速度偏慢，需后续调大 progress 系数；但本轮只改一个组件，暂不做。
- 若 gate 初始平均值低于 0.1，有效进步奖励过弱，可能停滞；但当前 sigma 设定在典型状态下 gate ≥ 0.3，应能维持足够的梯度。

---

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observations
    x_curr, y_curr = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]
    x_vel_next = next_obs[2]
    y_vel_next = next_obs[3]
    angle_next = next_obs[4]
    ang_vel_next = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 1. Progress reward: positive when moving closer to origin
    dist_curr = (x_curr**2 + y_curr**2) ** 0.5
    dist_next = (x_next**2 + y_next**2) ** 0.5
    progress = 10.0 * (dist_curr - dist_next)

    # 2. Landing gate (soft multiplicative constraint, replaces hard penalty)
    # Gaussian factors with relaxed sigma to keep gate >= 0.3 in moderate conditions
    sigma_vy = 0.5
    sigma_vx = 0.4
    sigma_angle = 0.2
    sigma_angvel = 0.4

    safe_vy = 2.718281828 ** (- (y_vel_next**2) / (sigma_vy**2))
    safe_vx = 2.718281828 ** (- (x_vel_next**2) / (sigma_vx**2))
    safe_angle = 2.718281828 ** (- (angle_next**2) / (sigma_angle**2))
    safe_angvel = 2.718281828 ** (- (ang_vel_next**2) / (sigma_angvel**2))

    landing_gate = safe_vy * safe_vx * safe_angle * safe_angvel

    # 3. Contact-based success reward (kept unchanged for now)
    contact_flag = min(left_contact, right_contact)  # 0.0 or 1.0

    sigma_vy_success = 0.2
    sigma_vx_success = 0.2
    sigma_angle_success = 0.1
    sigma_angvel_success = 0.2

    safe_vy_success = 2.718281828 ** (- (y_vel_next**2) / (sigma_vy_success**2))
    safe_vx_success = 2.718281828 ** (- (x_vel_next**2) / (sigma_vx_success**2))
    safe_angle_success = 2.718281828 ** (- (angle_next**2) / (sigma_angle_success**2))
    safe_angvel_success = 2.718281828 ** (- (ang_vel_next**2) / (sigma_angvel_success**2))

    contact_success_reward = 200.0 * contact_flag * safe_vy_success * safe_vx_success * safe_angle_success * safe_angvel_success

    # Progress is gated by landing soft constraints; contact reward is additive
    total_reward = landing_gate * progress + contact_success_reward

    components = {
        'progress': progress,
        'landing_gate': landing_gate,
        'contact_success_reward': contact_success_reward
    }

    return float(total_reward), components
```

# 诊断摘要
- **audit**: `soft_landing_penalty` 82% 份额驱动 agent 主动快速终止；需要消除常驻负奖励，改为乘性门控。
- **behavior**: agent 学会最快终止（len 501，100% terminated），可能主动坠毁以结束负奖励 episode。
- **signal**: `progress` 信号过于微弱，被惩罚完全淹没；`contact_success_reward` 极稀疏无法补偿。
- **level**: Level 2
- **hypothesis**: 移除惩罚并将其转化为 progress 的门控因子，消除“越快死越赚”的激励，agent 将恢复生存并逐步改善姿态以获得有效 progress，len 回升，得分转正。
- **risk**: progress 的 per‑step 量级较小，乘 gate 后学习速度可能偏慢，需后续加强 progress 系数。