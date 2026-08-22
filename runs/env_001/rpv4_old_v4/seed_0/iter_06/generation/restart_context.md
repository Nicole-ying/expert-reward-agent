# Fresh Restart Evidence

- target_score: 200.000
- best_score_so_far: -36.380

## Tried component structures

| structure | attempts | best_score | latest_score | status |
|---|---:|---:|---:|---|
| landing_proximity_reward + orientation_penalty + progress_reward + vel_damping | 3 | -36.380 | -59.190 | unsolved |
| fuel_penalty + landing_proximity_reward + orientation_penalty + progress_reward + vel_damping | 1 | -124.220 | -124.220 | unsolved |
| contact_reward + orientation_penalty + progress_reward + vel_damping | 1 | -159.490 | -159.490 | unsolved |

## Previous interventions

- iter 2 (score=-36.380, structure=landing_proximity_reward + orientation_penalty + progress_reward + vel_damping): selected_level: L2** — 修改组件的数学形态。contact_reward 完全死亡，调权重（L1）无法修复；必须改变触发/检测机制。同时保持同一语义方向（着陆成功信号），不是 L3 换框架。 | selected_intervention: 替换 `contact_reward` → 新建 `landing_proximity_reward`**：
- iter 3 (score=-434.870, structure=landing_proximity_reward + orientation_penalty + progress_reward + vel_damping): selected_level: L2** — 修改 `landing_proximity_reward` 的数学形态：从**纯状态型 Gaussian 乘积**改为**进度加权型（progress-weighted）**。这不是简单的系数调整（L1），也不是换框架（L3），而是改变同一组件的引导逻辑。 | selected_intervention: 目标组件**: `landing_proximity_reward`
- iter 4 (score=-59.190, structure=landing_proximity_reward + orientation_penalty + progress_reward + vel_damping): 现在我已掌握全局。让我整理诊断报告并给出修改方案。 | selected_level: L2** — 改变 `landing_proximity_reward` 的数学形态。这不是简单的系数调整（L1），因为 progress-weighted 形态本身有振荡套利的根本缺陷。也不是 L3，因为不需要换框架——状态高斯的基本思路（iter_2）方向正确，只是 σ 太宽。 | selected_intervention: 将 `landing_proximity_reward` 从 progress-weighted 形态改回状态高斯，但使用更紧的 σ（σ_dist=0.7 vs iter_2 的 2.0），并提高权重以补偿窄高斯。
- iter 5 (score=-124.220, structure=fuel_penalty + landing_proximity_reward + orientation_penalty + progress_reward + vel_damping): 现在我已掌握全局。让我整理诊断报告并给出修改方案。 | selected_level: L2** — 新增 `fuel_penalty` 组件（新语义，新数学形态），配合 `landing_proximity_reward` 权重下调（L1 级别的辅助调整）。两个修改服务于同一语义目标：「打破盘旋 exploit，驱动真正着陆」。 | selected_intervention: 1. **新增 `fuel_penalty`**：按动作类型惩罚发动机使用——无推力(action=0)奖励 0，姿态发动机(action=1/3)轻罚 -0.05，主发动机(action=2)重罚 -0.2。这使得盘旋（需要持续主发动机对抗重力）每步付出代价。

## Restart instruction

The previous search has stagnated. Propose a materially different design hypothesis, not merely a renamed or trivially rescaled copy.
Compare the tried structures and their scores before choosing the next direction.
If you continue a previous structure family, state what new evidence justifies it and change its mathematical mechanism or temporal semantics.
Expert skeletons are design primitives and risk hints, not a closed candidate list. You may combine, transform, or create a new signal using only declared environment inputs.
