# Fresh Restart Evidence

- target_score: 200.000
- best_score_so_far: -18.800

## Tried component structures

| structure | attempts | best_score | latest_score | status |
|---|---:|---:|---:|---|
| landing_approach_reward + progress + soft_landing_penalty | 1 | -18.800 | -18.800 | unsolved |
| contact_success_reward + landing_approach_reward + progress | 1 | -55.780 | -55.780 | unsolved |
| action_cost + angle_penalty + landing_soft_reward + progress + safety_penalty | 1 | -80.850 | -80.850 | unsolved |
| action_cost + gate_factor + shaping + success_bonus | 1 | -95.670 | -95.670 | unsolved |
| contact_success_reward + progress + soft_landing_penalty | 1 | -112.840 | -112.840 | unsolved |
| landing_bonus + progress + soft_landing_penalty | 1 | -115.300 | -115.300 | unsolved |
| contact_success_reward + landing_gate + progress | 1 | -115.490 | -115.490 | unsolved |
| action_cost + angle_penalty + boundary_penalty + landing_soft_reward + progress | 1 | -117.780 | -117.780 | unsolved |

## Previous interventions

- iter 7 (score=-80.850, structure=action_cost + angle_penalty + landing_soft_reward + progress + safety_penalty): 本轮修改 **Level 2 结构变换**，将永不死触发（active_rate=0）的 `boundary_penalty` 组件替换为一个新的 `safety_penalty` 组件，填补“碰撞前兆安全约束”的信号缺口。

## Restart instruction

The previous search has stagnated. Propose a materially different design hypothesis, not merely a renamed or trivially rescaled copy.
Compare the tried structures and their scores before choosing the next direction.
If you continue a previous structure family, state what new evidence justifies it and change its mathematical mechanism or temporal semantics.
Expert skeletons are design primitives and risk hints, not a closed candidate list. You may combine, transform, or create a new signal using only declared environment inputs.
