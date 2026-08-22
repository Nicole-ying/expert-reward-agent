# 设计理由

## 累积失败分析
- **Iter 1**: 直接使用 `forward` + `height_penalty` + `upright_penalty`，惩罚太重，agent几乎立即死亡（len≈59）。
- **Iter 2-3**: 引入 `forward_gated` + `health_gate` + `upright_penalty` + `z_velocity_penalty`，试图用门控和垂直速度惩罚增强稳定性，反而使存活时长跌得更惨（len≈244–319，score 为负）。
- **Iter 4**: 完全放弃惩罚，改用 `forward * height_factor` + `upright_bonus`（系数0.1）。存活时间飙升至 ∼990，但 **前进收益几乎被环境内置负奖励完全抵消**（生成 reward +0.67/step vs 环境奖励 –0.66/step，净收益≈0.01/step，累积 score 仅 69）。高度门控近乎饱和（height_factor≈0.87），直立奖励太弱（0.028/step），两者无法提供有效梯度。

**根因**：Iter 4 已经解决了“存活”问题，但 **主信号强度不足以克服环境负奖励**，导致策略卡在低速稳态，缺乏向上提升的激励。

## 重建策略——选择新骨架
我决定基于 **Iter 4 的“门控前进”框架**做**Level 3 重建**，而非另起全新的公式算子家族，原因：
- Iter 4 是唯一让 agent 活得足够长、且 **terminated 率低至 5%** 的骨架，说明其健康约束机制（height gate）是有效的。
- 累积记录显示引入惩罚往往导致死亡螺旋；但完全没有副约束又会放任低效运动。
- 新骨架的核心变动是 **大幅强化前向奖励强度**（克服环境负奖励），同时以 **极低权重的 hinge 惩罚** 抑制横向漂移和剧烈垂直抖动——这些惩罚仅当速度超过安全阈值时才激活，给足安全探索空间。

**已尝试但失败的路径**：
- 全局二次/线性惩罚（高度、直立、垂直速度）→ 存活优先级被压制，len 短。
- 过弱的前向信号（coeff=1.0）→ 被环境负奖励淹没。

**新骨架的本质不同**：
- **前向系数 ×2**，线性形态不变（避免凸化初期的速度震荡），但足以突破环境负奖励。
- **直立奖励系数 ×2**（0.1→0.2），增强姿态引导。
- **新增 hinge 惩罚**，仅对横向速度 >0.3 或垂直速度 >0.5 的部分征收极轻税，不触碰安全区。这与 Iter 2‑3 的全时二次惩罚截然不同，不会让 agent 因正常躯体摆动而受罚。

## 数学形式与系数校准
- **forward_reward** = `2.0 * body_x_velocity * height_factor`  
  期望 per‑step 贡献 ≈ 2.0 × 0.64 × 0.87 ≈ 1.11，加上直立和惩罚后，总生成奖励约 1.2/step，净收益约 +0.5/step，1000 步可累积 ≥500 分。
- **upright_bonus** = `0.2 * body_up_z`
- **lateral_penalty** = `-0.1 * max(0, |body_y_velocity| - 0.3)`  
  阈值 0.3 允许适度侧滑，超出部分轻罚。
- **vertical_penalty** = `-0.1 * max(0, |body_z_velocity| - 0.5)`  
  阈值 0.5 容忍正常步态中的重心起伏，仅压制过度弹跳。

所有惩罚的 per‑step 合计远小于主信号 (0.1‑0.2 << 1.1)，满足 `<0.5×主信号` 的负担限制。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- Extract observations ----------
    body_z = obs[0]
    quat_x = obs[2]
    quat_y = obs[3]
    body_x_velocity = obs[13]
    body_y_velocity = obs[14]
    body_z_velocity = obs[15]

    # ---------- Derived signals ----------
    # Uprightness: 1.0 = perfectly upright, -1.0 = upside-down
    body_up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)

    # ---------- Height gate (identical to iter4, proven effective) ----------
    height_low_danger = max(0.0, (0.30 - body_z) / 0.10)   # 0.30→0.20
    height_low_factor = max(0.0, 1.0 - height_low_danger)
    height_high_danger = max(0.0, (body_z - 0.90) / 0.10)  # 0.90→1.00
    height_high_factor = max(0.0, 1.0 - height_high_danger)
    height_factor = min(height_low_factor, height_high_factor)

    # ---------- Reward components ----------
    # Stronger forward propulsion to overcome environment's built-in penalty
    forward_reward = 2.0 * body_x_velocity * height_factor

    # Moderate upright guidance
    upright_bonus = 0.2 * body_up_z

    # Light hinge penalties — only activate beyond safe thresholds
    lateral_deviation = abs(body_y_velocity)
    lateral_penalty = -0.1 * max(0.0, lateral_deviation - 0.3)

    vertical_deviation = abs(body_z_velocity)
    vertical_penalty = -0.1 * max(0.0, vertical_deviation - 0.5)

    # ---------- Total reward ----------
    total_reward = forward_reward + upright_bonus + lateral_penalty + vertical_penalty

    components = {
        "forward": forward_reward,
        "upright_bonus": upright_bonus,
        "lateral_penalty": lateral_penalty,
        "vertical_penalty": vertical_penalty,
        "height_factor": height_factor,
    }

    return float(total_reward), components
```

# 诊断摘要
- **audit**: 高度终止边界已通过 height gate 软约束，前向速度是唯一与“持续移动”目标直接挂钩的信号，本次加大其系数；缺少速度效率信号已补。
- **behavior**: agent 能存活至截断但前进极慢，因为生成奖励刚好被环境负奖励抵消，净信号≈0。
- **signal**: forward 信号过弱（0.64/step），直立奖励近乎无效，高度门控失去梯度。
- **level**: Level 3（骨架微调本质属于重建，因核心系数和惩罚结构都变了）
- **hypothesis**: 加倍前向系数、轻量横向/垂直 hinge 惩罚将打破净零平衡，驱动更快的持续前进，同时保持高度安全，最终使 score 脱离 70 平台。
- **risk**: 若 2.0 系数仍不足以克服环境负奖励（或导致不稳定跌落），可能需要进一步凸化速度奖励；但当前设计保守且可后续调优。