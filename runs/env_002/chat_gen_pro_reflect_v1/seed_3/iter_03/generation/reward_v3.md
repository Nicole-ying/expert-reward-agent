# 诊断与修改

## 0. 信号覆盖审计
- **终止→前兆**：#5 终止条件为 `body_fallen_over`（摔倒）。#2 代码有 `balance_penalty`（倾斜角+角速度惩罚）作为前兆软信号，但 active_rate=100% 且 episode_sum_mean 仅 -4.95，惩罚力度极弱。`hull_angle` 二次惩罚在接近摔倒边界时梯度不足。
- **目标→进度**：#5 目标是"持续前进通过地形"。#2 有 `forward_progress` 直接给梯度，但被 `gait_quality` 门控大幅削减（forward_progress_raw=102.9 → forward_with_gait=85.6，损失 17%）。
- **效率信号**：#5 动作维度=4，未达≥6阈值，暂不要求 action penalty。
- **僵尸组件**：`both_off_ground_penalty` active_rate=39.7% 非僵尸，`gait_quality` active_rate=58.2% 偏低但仍有触发。
- **一句话结论**：`gait_quality` 门控在探索阶段过度压制前进奖励，导致 len 从 1600 暴跌至 194，机器人无法获得足够的前进梯度来学习行走。

---

## 1. 行为诊断
1. **agent 在做什么？** 快速失败。len=194.45（远低于 iter_1 的 1600），terminated=20/20（全部摔倒）。`forward_progress_raw` episode_sum_mean=102.9 但 len 只有 194，说明摔倒前的几步有速度但无法维持。**len 从 1600 断崖暴跌至 194，根因是 iter_2 将交替信号改为乘性门控**——机器人必须在步态协调的情况下才能获得前进奖励，而早期探索阶段步态必然是混乱的，门控几乎完全关闭了学习信号。

2. **干预哪个目标？** 主干预：**恢复前进奖励的可获取性**。`gait_quality` 门控的意图正确（消除原地踏步），但乘性门控对探索阶段过于严苛。需要改为**加性 bonus** 或**宽松门控**，确保机器人即使步态笨拙也能获得足够的前进梯度。

3. **这个方向还值得继续吗？** 仅 2 轮迭代，不算连续失败。但乘性门控方向导致 len 暴跌是明确的结构问题。**值得继续——用加性结构替代乘性结构**，保留步态引导但不压制主信号。

---

## 2. 选择干预层级
**Level 2 — 结构变换**：将 `forward_with_gait = forward_progress * gate` 改为 `forward_progress + gait_bonus`。步态质量变为加性 bonus（值域 0 ~ 1.0 per-step），不再作为门控。保持 `balance_penalty` 和 `both_off_ground_penalty` 不变。

**为什么选加性而非调整门控系数？** 当前 gate = `0.5 + 0.5 * gait_quality`，值域 [0.5, 1.0]。但 `gait_quality` active_rate 仅 58.2%，大量步骤步态混乱（两脚同时着地或同时腾空），门控在 0.5 低位运行。即使抬高底限至 0.8，也无法解决"步态混乱时奖励打折"的根本问题——打折意味着前进梯度减弱，而这正是探索阶段最需要梯度的时候。

---

## 3. 设计校准

### 当前组件 per-step 估算
- `forward_progress_raw` per-step ≈ 102.95 / 194.45 ≈ 0.529
- `gait_quality` per-step ≈ 113.15 / 194.45 ≈ 0.582
- `balance_penalty` per-step ≈ -4.95 / 194.45 ≈ -0.025
- `both_off_ground_penalty` per-step ≈ -15.45 / 194.45 ≈ -0.079

### 新设计
- 主信号 `forward_progress`：保持 `2.0 * max(0, horizontal_vel)^2`，per-step ≈ 0.53
- `gait_bonus`：`0.3 * gait_quality`，per-step ≈ 0.3 * 0.58 ≈ 0.17（约为主信号的 0.32x）
- 惩罚合计 per-step ≈ 0.025 + 0.079 = 0.104（约为主信号的 0.2x）✓
- 总惩罚负担 ≤ 主信号 0.5x ✓
- 新组件 `gait_bonus` ≤ 2x 主信号 ✓

---

# 设计理由
将 `gait_quality` 从**乘性门控**改为**加性 bonus**。core 前进奖励不再被步态质量打折，机器人即使步态混乱也能获得完整的前进梯度；协调步态时获得额外 0.3 * gait_quality（最高 0.3/step）的 bonus。这恢复了 iter_1 中 len=1600 时的探索能力，同时保留了步态质量的正向引导。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ============================================================
    # 1. forward_progress: 主学习信号（不受门控压制）
    #    w=2.0，凸化平方，拒绝负速度
    # ============================================================
    horizontal_vel = obs[2]
    forward_progress = 2.0 * (max(0.0, horizontal_vel) ** 2)

    # ============================================================
    # 2. gait_bonus: 步态交替质量作为加性奖励（替代乘性门控）
    #    使用 leg1_contact，leg2_contact 计算交替程度
    #    值域 [0, 1]，1 表示两脚接触状态完全相反
    #    bonus 系数 0.3，per-step ≤ 0.3，探索阶段也能获得正向引导
    # ============================================================
    leg1_contact = obs[8]
    leg2_contact = obs[13]
    gait_quality = 1.0 - abs(leg1_contact - leg2_contact)
    gait_bonus = 0.3 * gait_quality

    # ============================================================
    # 3. balance_maintenance: 防摔倒软约束
    #    惩罚倾斜角与角速度
    # ============================================================
    hull_angle = obs[0]
    hull_angular_vel = obs[1]
    balance_penalty = -0.5 * (hull_angle ** 2) - 0.1 * (hull_angular_vel ** 2)

    # ============================================================
    # 4. both_off_ground_penalty: 防止跳跃/双腿腾空
    # ============================================================
    contact_sum = leg1_contact + leg2_contact
    both_off_ground_penalty = -0.2 * (1.0 - contact_sum) if contact_sum < 0.5 else 0.0

    # 组合总奖励：前进主信号 + 步态加成 + 平衡/腾空惩罚
    total_reward = forward_progress + gait_bonus + balance_penalty + both_off_ground_penalty

    components = {
        "forward_progress": forward_progress,
        "gait_bonus": gait_bonus,
        "balance_penalty": balance_penalty,
        "both_off_ground_penalty": both_off_ground_penalty,
        "gait_quality": gait_quality  # 纯诊断
    }

    return float(total_reward), components
```

# 诊断摘要
- **audit**: gait_quality 乘性门控在探索阶段过度压制前进梯度，导致 len 从 1600 暴跌至 194
- **behavior**: 快速失败——机器人无法获得足够前进信号，几步后摔倒终止
- **signal**: 主信号 forward_progress 被门控阻断，需要恢复其完整可获取性
- **level**: Level 2
- **hypothesis**: 将步态质量从乘性门控改为加性 bonus 后，前进梯度不再被打折，机器人应恢复 iter_1 的探索长度（~1600），同时 0.3 * gait_quality 的正向引导仍会塑造协调步态
- **risk**: 最坏情况——机器人可能退化为 iter_1 的原地踏步策略（前进奖励可独立获取，无需理想步态）。但 iter_1 的 forward_progress per-step 仅 0.004 而当前为 0.53，说明速度奖励本身已有足够的"必须真正前进"的梯度，无需门控强制