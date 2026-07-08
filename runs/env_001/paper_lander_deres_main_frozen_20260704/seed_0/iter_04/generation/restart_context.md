# Fresh Restart Evidence

- target_score: 200.000
- best_score_so_far: -111.760

## Tried component structures

| structure | attempts | best_score | latest_score | status |
|---|---:|---:|---:|---|
| progress + soft_landing_bonus + stability_penalty | 2 | -111.760 | -115.610 | unsolved |

## Previous interventions

- iter 2 (score=-115.610, structure=progress + soft_landing_bonus + stability_penalty): selected_level**: Level 1——progress和stability_penalty同为逐步稠密信号，|penalty/progress|=0.858远超0.5经验触发器，且progress是唯一主引导信号。 | selected_intervention**: 仅降低stability_penalty三个系数各10倍：w_vel=0.001, w_angle=0.001, w_angvel=0.0005。其他组件不变。

## Restart instruction

The previous search has stagnated. Propose a materially different design hypothesis, not merely a renamed or trivially rescaled copy.
Compare the tried structures and their scores before choosing the next direction.
If you continue a previous structure family, state what new evidence justifies it and change its mathematical mechanism or temporal semantics.
Expert skeletons are design primitives and risk hints, not a closed candidate list. You may combine, transform, or create a new signal using only declared environment inputs.
