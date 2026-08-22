# Fresh Restart Evidence

- target_score: 200.000
- best_score_so_far: -27.460

## Tried component structures

| structure | attempts | best_score | latest_score | status |
|---|---:|---:|---:|---|
| contact_bonus + goal_proximity + orientation_penalty + velocity_penalty | 3 | -27.460 | -27.460 | unsolved |
| goal_proximity + landing_progress + orientation_penalty + velocity_penalty | 1 | -111.620 | -111.620 | unsolved |
| crash_penalty + progress_reward + touchdown_bonus | 1 | -113.330 | -113.330 | unsolved |
| crash_penalty + landing_reward + progress_reward | 2 | -307.970 | -307.970 | unsolved |

## Previous interventions

- iter 4 (score=-111.620, structure=goal_proximity + landing_progress + orientation_penalty + velocity_penalty): selected_level**：**L2: 理由：需要改变 contact_bonus 的数学形态——从稠密状态奖励改为“接触事件 + 一次性成功着陆事件”，属于 L2 语义修改。权重会适当微调，但核心是公式形态变化。 | selected_intervention**：
- iter 7 (score=-113.330, structure=crash_penalty + progress_reward + touchdown_bonus): `selected_level`：**L2: 证据表明当前的主信号框架（持续 proxy）被 exploit，必须改变数学形态：移除持续的 `landing_proxy`，替换为一次性接触事件奖励。这不是简单调参（L1），而是语义级别的组件替换，但又未到全盘推倒主框架（L3），因为接近目标的方向信号（progress）仍然有价值。 | `selected_intervention`：

## Restart instruction

The previous search has stagnated. Propose a materially different design hypothesis, not merely a renamed or trivially rescaled copy.
Compare the tried structures and their scores before choosing the next direction.
If you continue a previous structure family, state what new evidence justifies it and change its mathematical mechanism or temporal semantics.
Expert skeletons are design primitives and risk hints, not a closed candidate list. You may combine, transform, or create a new signal using only declared environment inputs.
