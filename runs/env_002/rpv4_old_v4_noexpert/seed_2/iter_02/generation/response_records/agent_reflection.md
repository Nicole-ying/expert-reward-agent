# Response Record

# 诊断报告

## evidence
- 当前唯一一轮 iter_1：`score=233.56`（目标 300），`len=1092.0`，`terminated=20/20`，无 truncation。
- 组件最终占比：`forward_progress_reward` magnitude_share **98.5%**（ep_sum=412.14），`upright_stability_penalty` 仅 1.5%（ep_sum=-6.07）。奖励几乎被单一前进速度信号垄断。
- **关键不一致**：组件 ep_sum 之和 ≈ `412.14 - 6.07 = 406.07`，但官方评估 `score=233.56`。同一批评估 episode 上，我们的代理奖励总和远高于外部得分。
- `score_range=[-43.73, 290.03]`：最好的单集已到 290（离 300 仅 3%），说明策略“有能力”逼近目标；均值被拉低，说明存在低分尾部（可能的高力矩/低效率或偶发失稳）。
- checkpoint 趋势：forward_progress_reward 从 cp10% 的 8.6 一路升到 cp100% 的 426.8，整体 score 从 -97.7 升到 247.5，说明前向推进信号有效且在持续被学习。
- 环境卡明确列出 `energy_efficiency_penalty` 为条件职责：`action[0..3]` 力矩可用于能耗惩罚，且任务目标原文包含“尽量减少能量消耗”。当前奖励**完全没有能量/力矩成本信号**。

## behavior_diagnosis
策略已经学会稳定前向行走（平均 1092 步、upright 惩罚很小且随训练下降），前向速度代理信号很高。但从“代理奖励 406 vs 外部得分 233”的缺口看，策略大概率在用**过量关节力矩/低能效步态**换取前进速度；外部指标很可能扣除了能耗成本，而我们没有把这一成本教给策略。行为不是摔倒式失败，而是**高能耗、高力矩抖动**的次优前进，导致官方分被能耗拖累。

## signal_completeness
必要职责：
- `forward_progress` ✅（有，且有效）
- `upright_stability` ✅（有，且被学习抑制）
- `energy_efficiency` ❌（缺失）——这是任务目标中明确声明的次级目标，当前奖励完全未覆盖；同时环境卡把 `excessive_hip_knee_torque_jitter` 列为失败风险，缺失力矩成本会让策略没有动力去平滑控制。

## selected_level
**L2**（新增一个组件的数学形态）。理由：不是单纯调系数，而是补上一个缺失的语义职责（能耗惩罚）。当前分数 233 离目标 300 仍有 22% 差距，不满足“<10% 强制 L1”的条件；且本轮不是“上一轮改善 >30 分”的延续，没有必须只做 L1 的约束。

## selected_intervention
- 目标组件：新增 `energy_penalty`，基于四个关节力矩的 L2 范数：`-energy_weight * sum(action_i**2)`。
- 权重 `energy_weight = 0.05`，保持轻量，避免压制前向探索；预计让它在 magnitude_share 中占次要位置（远小于 forward_progress）。
- 其余组件（forward_progress_reward、upright_stability_penalty）**保持原样**，避免大规模重写丢弃已验证有效的前进/稳定信号。

## falsifiable_hypothesis
如果外部得分确实包含能耗成本（由“组件和 406 vs 官方分 233”的缺口支持），那么在训练奖励中加入轻量力矩成本，会让策略学会用更小、更平滑的力矩维持前进，从而在几乎不损失前进速度的情况下降低能耗扣分，使官方分向 300 靠拢。可验证：下一轮 `energy_penalty` 会有非零 active_rate 且 magnitude_share 处于次要区间（约 5%–20%），`forward_progress_reward` 不会塌缩，整体 score 上升。

## expected_next_round
策略仍保持稳定前进步态，前向速度基本不降或微降；关节力矩幅度与抖动下降，能耗更优；官方得分预计从 233 向 260–290 区间提升，低分尾部 episode 减少。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # --- forward progress (primary) ---
    forward_weight = 1.0
    speed_scale = 1.0
    forward_speed = 0.5 * (obs[2] + next_obs[2])
    forward_progress_reward = (
        forward_weight
        * forward_speed
        / ((forward_speed ** 2 + speed_scale ** 2) ** 0.5)
    )

    # --- upright stability (health constraint) ---
    angle_deadband = 0.5
    angvel_deadband = 1.5
    angle_weight = 1.0
    angvel_weight = 0.05
    eps = 1e-6

    avg_angle = 0.5 * (obs[0] + next_obs[0])
    avg_angvel = 0.5 * (obs[1] + next_obs[1])

    angle_raw = abs(avg_angle) - angle_deadband
    angle_excess = 0.5 * (angle_raw + (angle_raw ** 2 + eps) ** 0.5)
    angle_penalty = angle_weight * angle_excess ** 2

    angvel_raw = abs(avg_angvel) - angvel_deadband
    angvel_excess = 0.5 * (angvel_raw + (angvel_raw ** 2 + eps) ** 0.5)
    angvel_penalty = angvel_weight * angvel_excess ** 2

    upright_stability_penalty = -(angle_penalty + angvel_penalty)

    # --- energy efficiency (secondary, previously missing) ---
    energy_weight = 0.05
    torque_sq = action[0] ** 2 + action[1] ** 2 + action[2] ** 2 + action[3] ** 2
    energy_penalty = -energy_weight * torque_sq

    total_reward = forward_progress_reward + upright_stability_penalty + energy_penalty

    components = {
        "forward_progress_reward": float(forward_progress_reward),
        "upright_stability_penalty": float(upright_stability_penalty),
        "energy_penalty": float(energy_penalty)
    }

    return float(total_reward), components
```
