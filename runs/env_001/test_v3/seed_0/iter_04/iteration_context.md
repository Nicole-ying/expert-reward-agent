# Iteration Context

## Recommended Action
**tune** — 当前骨架仅运行1轮，得分158.82接近目标200，但landing_shaping主导（均值1.625），progress_reward过弱（均值0.204），可能导致agent在目标附近震荡而不完成着陆。建议调大progress_reward系数（如100）并降低landing_shaping系数（如2.0），使progress成为主信号。

## Agent Memory
| iter | score | best | skeleton_summary | trend |
|------|-------|------|------------------|-------|

## Expert Cards
## stability_penalty_dominance
- signal: abs(stability_penalty_mean) > abs(progress_reward_mean)
- risk: agent may become overly conservative or afraid to move
- fix: reduce angle/angular-velocity penalty or make it conditional near target

## goal_near_oscillation
- signal: distance/progress improves but episode length is long and landing proxy rarely triggers
- risk: agent hovers or oscillates around the goal without completing the task
- fix: add smooth low-speed, low-angle, near-target shaping; avoid pure distance reward

## Training Evidence
# Training Feedback

## External evaluation
- score: 158.822193
- episode_length: 728.400000 (mean)
- range: [93.826443, 208.689203]
- errors: 0

## Component evidence

| component | mean | abs_mean | nonzero_rate | min | max |
|-----------|------|----------|-------------|-----|-----|
| landing_shaping | 1.624923 | 1.624923 | 0.542595 | 0.000000 | 4.990018 |
| progress_reward | 0.203996 | 0.219075 | 0.998808 | -1.641093 | 2.296980 |
| stability_penalty | -0.037143 | 0.037143 | 1.000000 | -0.456914 | -0.000000 |
| total_reward | 1.791776 | 1.809284 | 1.000000 | -1.843835 | 4.990014 |
| generated_reward | 1.791776 | 1.809284 | 1.000000 | -1.843835 | 4.990014 |
| original_env_reward | -0.172914 | 1.689840 | 1.000000 | -100.000000 | 136.593785 |

## Signals
partial_progress; penalty_dominance:landing_shaping; penalty_dominance:generated_reward; penalty_dominance:original_env_reward
