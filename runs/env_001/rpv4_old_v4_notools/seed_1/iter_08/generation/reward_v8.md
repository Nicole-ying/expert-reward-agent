# 诊断报告

## evidence
- **iter_7 得分**：score=-51.02，范围为[-101.3, -24.2]；terminated=20/20，len=72.4，说明所有 episode 均提前终止（可能坠毁或成功），但成功率约 35%（bonus avg=70，每次+200）。
- **组件 magnitude_share**：`soft_landing_bonus` 占 71.6%，但其 active_rate 仅 0.5%，极其稀疏但主导奖励，一旦触发会剧烈扰动价值估计，训练不稳定。
- **checkpoint 趋势**：cp40% 时 `soft_landing_bonus` 均值骤升至 960（即约 4.8 次成功/20ep），同期 `touchdown_reward` 升至 127，外部得分 -10.7 为历史最佳，但随后退化到 -50~-60，表明暴发性成功事件导致策略崩溃。
- **速度惩罚与任务冲突**：`speed_penalty` 全局惩罚一切速度（vx²+vy²），与“尽快降落”目标矛盾，agent 被迫在快速下降与安全减速间挣扎，早期探索时被过度约束。
- **touchdown_reward 乘积形态**：quality 为三个因子相乘，对不完全满足条件的状态梯度极小，active_rate 仅 0.6%，缺乏有效的局部引导。
- **缺少持续着陆信号**：目前只有一次性事件奖励，没有对保持成功状态（双腿接触、低速度、低角度、近原点）的每步鼓励，导致成功状态不稳定。

## behavior_diagnosis
策略处于“粗放着陆”模式：能偶尔触发成功事件（约 35% episode），但大部分时间因姿态或速度控制不当而提前坠毁，得分大幅为负。训练过程中奖励函数的不稳定信号（稀疏大额 bonus）使得 agent 在学到高成功率的局部最优后迅速崩溃，无法固化稳定着陆策略。

## signal_completeness
- **缺少密集着陆质量信号**：成功条件满足时应有持续的每步正奖励，而非一次性脉冲。
- **缺少阶段平衡**：早期下降阶段不应被速度惩罚抑制，晚期接近地面时需要强力减速约束。
- **touchdown_reward 梯度不足**：乘积形态导致不满足所有条件时奖励接近零，难以提供渐进式引导。

## selected_level
**L2** — 需要修改数学形态，而不仅仅是调参。稀疏事件奖励改为密集每步奖励、速度惩罚引入阶段门控、touchdown quality 从乘积改为加和，这些都不是单纯系数调整。

## selected_intervention
1. **移除 `soft_landing_bonus`**，替换为 `landing_quality_reward`——成功条件满足时每步给 +10，持续奖励稳定着陆，消除大额脉冲。
2. **`speed_penalty` 引入阶段门控**：当 y < 0.5（接近地面）时采用强惩罚 `-0.1*(vx²+vy²)`，否则弱惩罚 `-0.01*(vx²+vy²)`，允许早期快速下降，晚期强制减速。
3. **`touchdown_reward` 质量函数改为加和**：`quality = max(0,1-dist/0.3) + max(0,1-speed/0.3) + max(0,1-|angle|/0.2)`，每个因子独立提供梯度，避免乘性塌缩。

## falsifiable_hypothesis
- 密集的 `landing_quality_reward` 使成功状态的价值估计平滑，训练稳定性提升，不会出现 cp40% 式的爆发-崩溃循环。
- 阶段门控速度惩罚解开早期速度约束，让 agent 更快到达目标区域，同时保留末端安全减速，整体 episode 长度下降，得分改善。
- 加和式 touchdown 质量提供更渐进的接触-稳定引导，active_rate 上升，学习加速。

## expected_next_round
得分应突破 -30 分（当前最佳 -10 但不可再现，预期下一轮至少达到 -20 以上），`landing_quality_reward` active_rate 上升至 2-5%，`touchdown_reward` 大小和 active_rate 均增加，episode 长度缩短，训练曲线更平滑。

---

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Components:
      proximity_progress  : dense - progress towards origin (target at (0,0))
      attitude_penalty    : penalty for tilt and angular velocity
      speed_penalty       : stage‑gated penalty on linear velocity (loose when high, strict near ground)
      touchdown_reward    : additive quality reward for any leg contact
      landing_quality_reward : per‑step reward while fully stable on the pad (replaces one‑time bonus)
    """
    # ---------- proximity progress ----------
    old_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    new_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    proximity_progress = 20.0 * (old_dist - new_dist)

    # ---------- attitude penalty ----------
    angle = next_obs[4]
    angular_vel = next_obs[5]
    attitude_penalty = -0.6 * (angle ** 2) - 0.12 * (angular_vel ** 2)

    # ---------- speed penalty (stage‑gated) ----------
    y = next_obs[1]                   # vertical coordinate, decreasing towards ground
    vx = next_obs[2]
    vy = next_obs[3]
    speed_sq = vx ** 2 + vy ** 2
    if y < 0.5:                       # near ground – strict braking
        speed_penalty = -0.1 * speed_sq
    else:                             # in flight – allow faster descent
        speed_penalty = -0.01 * speed_sq

    # ---------- touchdown reward (leg contact, additive quality) ----------
    leg1 = next_obs[6]
    leg2 = next_obs[7]
    leg_contact = (leg1 + leg2) > 0.5    # at least one leg touching
    if leg_contact:
        dist = new_dist
        speed = speed_sq ** 0.5
        ang = abs(angle)
        # additive factors: each independently rewards closeness to target conditions
        q_dist  = max(0.0, 1.0 - dist / 0.3)
        q_speed = max(0.0, 1.0 - speed / 0.3)
        q_angle = max(0.0, 1.0 - ang / 0.2)
        quality = q_dist + q_speed + q_angle   # range 0..3
        touchdown_reward = 3.0 * quality
    else:
        touchdown_reward = 0.0

    # ---------- landing quality (per‑step stable‑landing reward) ----------
    leg_contact_stable = (next_obs[6] + next_obs[7]) > 1.5      # both legs firmly
    pos_ok   = abs(next_obs[0]) < 0.3
    speed_ok = speed_sq ** 0.5 < 0.3
    angle_ok = abs(angle) < 0.2
    if leg_contact_stable and pos_ok and speed_ok and angle_ok:
        landing_quality_reward = 10.0
    else:
        landing_quality_reward = 0.0

    # ---------- total ----------
    total_reward = (proximity_progress
                    + attitude_penalty
                    + speed_penalty
                    + touchdown_reward
                    + landing_quality_reward)

    components = {
        "proximity_progress": proximity_progress,
        "attitude_penalty": attitude_penalty,
        "speed_penalty": speed_penalty,
        "touchdown_reward": touchdown_reward,
        "landing_quality_reward": landing_quality_reward,
    }
    return float(total_reward), components
```