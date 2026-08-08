# Fresh Restart Evidence

- target_score: 2000.000
- best_score_so_far: 67.710

## Tried component structures

| structure | attempts | best_score | latest_score | status |
|---|---:|---:|---:|---|
| forward_reward + height_reward + upright_reward | 1 | 67.710 | 67.710 | unsolved |
| forward_reward + height_penalty + lateral_penalty + upright_penalty | 1 | -5.090 | -5.090 | unsolved |
| gated_forward + height_gate + lateral_penalty + upright_reward | 1 | -37.130 | -37.130 | unsolved |
| action_penalty + forward_gated_height | 1 | -55.540 | -55.540 | unsolved |
| action_penalty + forward_gated + height_reward | 1 | -112.410 | -112.410 | unsolved |
| forward_gated + height_reward | 1 | -271.100 | -271.100 | unsolved |
| gated_forward + height_gate + joint_vel_penalty + lateral_penalty + upright_reward | 1 | -353.940 | -353.940 | unsolved |

## Previous interventions

- No structured intervention fields were available in the historical responses.

## Restart instruction

The previous search has stagnated. Propose a materially different design hypothesis, not merely a renamed or trivially rescaled copy.
Compare the tried structures and their scores before choosing the next direction.
If you continue a previous structure family, state what new evidence justifies it and change its mathematical mechanism or temporal semantics.
Expert skeletons are design primitives and risk hints, not a closed candidate list. You may combine, transform, or create a new signal using only declared environment inputs.
