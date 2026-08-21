## 1. 诊断报告

**evidence**
- iter_1 最终得分 278.99 / 目标 300，差距 21 分（约 7%），已经很接近目标。
- `terminated=20/20`、`len=1275.6`：回合普遍较长，说明策略已经能稳定前进、大概率走到地形尽头，不是摔倒/原地抖动模式。
- 组件构成：`forward_progress` ep_sum=697.4，magnitude_share=99.1%；`upright_stability` ep_sum=-6.7，magnitude_share=0.9%。前向信号是唯一实质驱动力，且工作正常；直立惩罚全程接近 0，说明躯干已经很稳定，不是当前瓶颈。
- 当前奖励函数**完全没有能量/力矩项**。而环境目标明确写有「次目标：降低能量消耗」，且官方评估（本 benchmark 的 score）在 BipedalWalker 类任务里通常对关节力矩有惩罚项。
- score_range=[36.3, 294.4]：多数回合接近满分，少数回合差，说明步态本身可行但可能伴随不必要的力矩消耗。

**behavior_diagnosis**
策略当前学到的是「稳定但可能高力矩的步态」：它能持续前进并基本走完地形（健康的主目标行为），但因为没有力矩成本信号，策略没有动力去节省关节力矩。这不是病态行为，而是缺少次目标引导，导致在官方 score 上被力矩惩罚扣分，卡在 279 而非 300。

**signal_completeness**
- 前向进度：✓ 已有，且是主导信号。
- 直立/稳定：✓ 已有，虽然强度很小，但 agent 已经稳定，不是缺口。
- 能量效率：✗ **缺失**。这是当前唯一缺失的职责，且环境明确将其列为次目标（conditional role：当已出现有效前进时应加入）。

**selected_level**
L2（新增一个组件）。理由：现有两个组件都不涉及动作力矩，纯 L1 调系数无法引入能量效率目标。虽然接近目标时应优先 L1，但这里 L1 无对应杠杆；新增项采用很小的系数、有界形式，语义上只是补一个温和的次目标，不重写已经有效的 forward_progress / upright_stability 公式。

**selected_intervention**
新增组件 `energy_efficiency_penalty`：`-0.03 * sum(action_i^2)`。保持 `forward_progress` 和 `upright_stability` 完全不变。系数 0.03 使典型力矩水平下的惩罚约为前向奖励的 10% 左右，足以引导省力步态，但远不足以压过前向主信号。

**falsifiable_hypothesis**
如果加入温和力矩惩罚后，策略在保持前进距离/速度的同时降低关节力矩，官方 score 的力矩惩罚项会下降，得分应从 279 向 300 靠拢；`forward_progress` 仍应保持主导（magnitude_share 仍 > 85%）。若下一轮出现前进速度明显下降、回合变短或 score 反而下降，则说明 0.03 过强，应回退为 L1 微调（降系数）。

**expected_next_round**
策略预计学到更省力的步态：前进距离/速度基本保持，关节力矩均值下降，`energy_efficiency_penalty` 活跃率 100% 且 magnitude_share 在 5%~15% 之间，`forward_progress` 仍为最大份额；score 预计小幅上升至 285~300 区间。

## 2. 完整 Python 代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：前向速度，使用光滑有界形式
    horizontal_velocity = next_obs[2]
    forward_progress = 2.0 * horizontal_velocity / (1.0 + abs(horizontal_velocity))

    # 稳定性约束：惩罚躯干倾角与躯干角速度
    torso_angle = next_obs[0]
    torso_angle_sq = torso_angle * torso_angle
    angle_penalty = 1.5 * torso_angle_sq / (1.0 + torso_angle_sq)

    torso_angular_velocity = next_obs[1]
    torso_angular_velocity_sq = torso_angular_velocity * torso_angular_velocity
    angular_velocity_penalty = 0.5 * torso_angular_velocity_sq / (1.0 + torso_angular_velocity_sq)

    upright_stability = -(angle_penalty + angular_velocity_penalty)

    # 能量效率（次目标）：温和惩罚关节力矩平方和，引导省力步态。
    # action 范围为 [-1, 1] x 4，sum(action^2) 有界于 [0, 4]。
    action_sq = sum(a * a for a in action)
    energy_efficiency_penalty = -0.03 * action_sq

    total_reward = forward_progress + upright_stability + energy_efficiency_penalty

    components = {
        "forward_progress": forward_progress,
        "upright_stability": upright_stability,
        "energy_efficiency_penalty": energy_efficiency_penalty,
    }

    return float(total_reward), components
```