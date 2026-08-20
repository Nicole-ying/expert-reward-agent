# Fresh Restart Evidence

- target_score: 200.000
- best_score_so_far: -37.030

## Tried component structures

| structure | attempts | best_score | latest_score | status |
|---|---:|---:|---:|---|
| angular_vel_penalty + landing + progress + speed_penalty | 4 | -37.030 | -197.460 | unsolved |
| angular_vel_penalty + progress + speed_penalty + velocity_guidance | 1 | -295.330 | -295.330 | unsolved |

## Previous interventions

- iter 2 (score=-37.030, structure=angular_vel_penalty + landing + progress + speed_penalty): 现在我已经掌握了完整的画面。让我整理诊断并制定修改方案。 | selected_level: L2** — 修改 `landing` 组件的数学形态：从"严格乘积门控（5 个因子全部非零才有奖励）"改为"基于 proximity 门控的连续求和公式，提供部分梯度"。这不是简单调权重（L1），也不是换框架（L3）。 | selected_intervention: 目标组件**：`landing`（着陆奖励）
- iter 3 (score=-222.850, structure=angular_vel_penalty + landing + progress + speed_penalty): 现在我已掌握完整画面。以下是诊断和修改方案。 | selected_level: L2** — 需要改变 landing 的数学形态（从 proximity-gated 改为 contact-gated），单纯调权重（L1）无法解决根本逻辑缺陷：无论权重多低，只要 proximity > 0 就给 reward，agent 就会 exploit。 | selected_intervention: 目标组件**：`landing`
- iter 4 (score=-295.330, structure=angular_vel_penalty + progress + speed_penalty + velocity_guidance): 现在我已经掌握了所有关键信息。让我整理诊断并制定修改方案。 | selected_level: L2**：替换一个组件的数学形态/语义。不是调系数（L1），也不是全框架重构（L3）。 | selected_intervention: 目标组件：** 移除 `landing`，新增 `velocity_guidance`。
- iter 5 (score=-197.460, structure=angular_vel_penalty + landing + progress + speed_penalty): selected_level: L2** — 替换一个组件的数学形态。将 `velocity_guidance`（方向对齐）替换为 `landing`（proximity-gated + contact-amplified 着陆引导）。这不是调参（L1），也不是换整个框架（L3）。这是将"方向引导"的语义改为"目标接近+着陆质量引导"。 | selected_intervention: 移除 `velocity_guidance`，新增 `landing` 组件**：

## Restart instruction

The previous search has stagnated. Propose a materially different design hypothesis, not merely a renamed or trivially rescaled copy.
Compare the tried structures and their scores before choosing the next direction.
If you continue a previous structure family, state what new evidence justifies it and change its mathematical mechanism or temporal semantics.
Expert skeletons are design primitives and risk hints, not a closed candidate list. You may combine, transform, or create a new signal using only declared environment inputs.
