# Iteration Context

## Recommended Action
**tune** — 当前骨架仅运行1轮，得分-102.17远低于目标200。stability_penalty均值-0.134远大于progress_delta_reward均值-0.005，导致agent不敢移动。best_reward与current代码完全相同，因此无需revert。建议降低stability_penalty系数（如将angle_penalty从-1.0降至-0.5），并提高progress_delta_reward系数（如从2.0增至5.0），以平衡探索与稳定。

## Agent Memory
| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | energy_penalty + progress_delta_reward + stability_penalty | -102.17 | -102.17 | 0.00 | 33.00 | energy_penalty=-0.016 progress_delta_reward=-0.005 stability_penalty=-0.134 | new_best |

## Expert Cards
## stability_penalty_dominance
- signal: abs(stability_penalty_mean) > abs(progress_reward_mean)
- risk: agent may become overly conservative or afraid to move
- fix: reduce angle/angular-velocity penalty or make it conditional near target

## generated_reward_negative_average
- signal: total generated reward mean is negative during most training steps
- risk: agent receives mostly punitive feedback and may not discover useful behavior
- fix: rebalance progress and penalty terms; avoid large always-on penalties in early versions

## Stable Lessons (from previous iterations)
- Target: mean external score >= 200.
- terminal_success_reward and terminal_failure_penalty blocked (no explicit flag).
- Use external evaluation as fitness signal, not generated_reward alone.
- Contact only inside guarded landing proxy (near target + low speed + stable angle).
- Prefer continuous shaping over hard sparse bonuses.

## Component Evidence
| component | mean | abs_mean | nonzero_rate | min | max |
|-----------|------|----------|-------------|-----|-----|
| energy_penalty | -0.015771 | 0.015771 | 1.000000 | -0.040000 | -0.000012 |
| progress_delta_reward | -0.005342 | 0.036760 | 1.000000 | -0.478311 | 0.418580 |
| stability_penalty | -0.134152 | 0.134152 | 1.000000 | -3.276569 | -0.000147 |
| total_reward | -0.155266 | 0.156463 | 1.000000 | -3.385644 | 0.279435 |
| generated_reward | -0.155266 | 0.156463 | 1.000000 | -3.385644 | 0.279435 |
| original_env_reward | -1.959425 | 1.981518 | 1.000000 | -100.000000 | 0.602193 |
