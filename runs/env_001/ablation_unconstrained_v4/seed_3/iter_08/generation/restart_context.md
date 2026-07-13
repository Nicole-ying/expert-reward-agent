# Fresh Restart Evidence

- target_score: 200.000
- best_score_so_far: -10.160

## Tried component structures

| structure | attempts | best_score | latest_score | status |
|---|---:|---:|---:|---|
| contact_reward + gated_goal_reward | 1 | -10.160 | -10.160 | unsolved |
| contact_reward + engine_penalty + goal_reward + height_reward + landing_bonus | 1 | -20.700 | -20.700 | unsolved |
| contact_bonus + engine_penalty + gated_goal + height_reward | 1 | -63.810 | -63.810 | unsolved |
| global_speed_penalty + progress_reward + soft_landing | 1 | -104.190 | -104.190 | unsolved |
| engine_penalty + landing_reward + lateral_penalty + progress_reward | 2 | -122.030 | -122.030 | unsolved |
| contact_reward + descent_reward + engine_penalty + landing_bonus | 1 | -123.190 | -123.190 | unsolved |

## Previous interventions

- iter 2 (score=-63.810, structure=contact_bonus + engine_penalty + gated_goal + height_reward): 修改方案：: 1. **强化接触奖励**：增加双腿接触的奖励权重，并额外对双腿同时接触给予大幅奖励，确保Agent有动力接触垫。
- iter 4 (score=-123.190, structure=contact_reward + descent_reward + engine_penalty + landing_bonus): 修改方案：: 用线性下降奖励（最高限制）替代倒数高度奖励，驱动agent持续下降。
- iter 6 (score=-122.770, structure=engine_penalty + landing_reward + lateral_penalty + progress_reward): 修改方案: 主奖励：`progress_reward` = `w_progress * (dist_now - dist_next)`，乘以安全门控因子。

## Restart instruction

The previous search has stagnated. Propose a materially different design hypothesis, not merely a renamed or trivially rescaled copy.
Compare the tried structures and their scores before choosing the next direction.
If you continue a previous structure family, state what new evidence justifies it and change its mathematical mechanism or temporal semantics.
Expert skeletons are design primitives and risk hints, not a closed candidate list. You may combine, transform, or create a new signal using only declared environment inputs.
