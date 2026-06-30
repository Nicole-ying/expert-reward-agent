# Iteration Context

## Recommended Action
**tune** — stability_penalty dominates (mean -0.340, nonzero 1.0) causing early failure/crash; progress_reward too weak (mean 0.160) to counteract; soft_landing_bonus too sparse (0.5% trigger). Recommend reducing stability_penalty coefficient and increasing progress_reward coefficient.

## Agent Memory
| iter | score | best | skeleton_summary | trend |
|------|-------|------|------------------|-------|

## Expert Cards
## stability_penalty_dominance
- signal: abs(stability_penalty_mean) > abs(progress_reward_mean)
- risk: agent may become overly conservative or afraid to move
- fix: reduce angle/angular-velocity penalty or make it conditional near target

## early_failure_or_crash
- signal: negative external score and short episode length
- risk: reward does not guide stable control before termination
- fix: add smooth approach/landing signals; avoid relying on sparse terminal-like proxy

## Training Evidence
# Training Feedback

## External evaluation
- score: -111.262458
- episode_length: 74.100000 (mean)
- range: [-123.713417, -96.378727]
- errors: 0

## Component evidence

| component | mean | abs_mean | nonzero_rate | min | max |
|-----------|------|----------|-------------|-----|-----|
| progress_reward | 0.160497 | 0.169886 | 0.999990 | -0.420961 | 0.423601 |
| soft_landing_bonus | 0.010589 | 0.010589 | 0.005294 | 0.000000 | 2.000000 |
| stability_penalty | -0.339930 | 0.339930 | 1.000000 | -2.804435 | -0.000001 |
| total_reward | -0.168845 | 0.188854 | 1.000000 | -2.964972 | 2.006727 |
| generated_reward | -0.168845 | 0.188854 | 1.000000 | -2.964972 | 2.006727 |
| original_env_reward | -1.556501 | 2.357025 | 1.000000 | -100.000000 | 112.994905 |

## Signals
early_failure_or_crash; sparse_proxy:soft_landing_bonus; penalty_dominance:stability_penalty; penalty_dominance:generated_reward; penalty_dominance:original_env_reward
