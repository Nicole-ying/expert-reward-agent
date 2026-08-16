# Fresh Restart Evidence

- target_score: 200.000
- best_score_so_far: -27.460

## Tried component structures

| structure | attempts | best_score | latest_score | status |
|---|---:|---:|---:|---|
| contact_bonus + goal_proximity + orientation_penalty + velocity_penalty | 3 | -27.460 | -27.460 | unsolved |
| goal_proximity + landing_progress + orientation_penalty + velocity_penalty | 1 | -111.620 | -111.620 | unsolved |

## Previous interventions

- iter 4 (score=-111.620, structure=goal_proximity + landing_progress + orientation_penalty + velocity_penalty): selected_level**：**L2: 理由：需要改变 contact_bonus 的数学形态——从稠密状态奖励改为“接触事件 + 一次性成功着陆事件”，属于 L2 语义修改。权重会适当微调，但核心是公式形态变化。 | selected_intervention**：

## Restart instruction

The previous search has stagnated. Propose a materially different design hypothesis, not merely a renamed or trivially rescaled copy.
Compare the tried structures and their scores before choosing the next direction.
If you continue a previous structure family, state what new evidence justifies it and change its mathematical mechanism or temporal semantics.
Expert skeletons are design primitives and risk hints, not a closed candidate list. You may combine, transform, or create a new signal using only declared environment inputs.
