# Iteration Context

## Recommended Action
**tune** — 外部得分-111.79远低于目标200，且两轮无改善。progress_reward均值0.242弱，stability_penalty均值-0.331主导，导致penalty dominance。soft_landing_bonus触发率仅0.5%，稀疏无效。建议调大progress_reward系数至50以上，减小stability_penalty系数至0.1以下，并提高soft_landing_bonus触发条件或改为连续shaping。

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
- score: -111.792476
- episode_length: 74.100000 (mean)
- range: [-121.129284, -96.806702]
- errors: 0

## Component evidence

| component | mean | abs_mean | nonzero_rate | min | max |
|-----------|------|----------|-------------|-----|-----|
| progress_reward | 0.241773 | 0.255777 | 0.999998 | -0.608839 | 0.634183 |
| soft_landing_bonus | 0.010212 | 0.010212 | 0.005106 | 0.000000 | 2.000000 |
| stability_penalty | -0.330620 | 0.330620 | 1.000000 | -1.469923 | -0.000000 |
| total_reward | -0.078635 | 0.110697 | 1.000000 | -1.813497 | 2.002875 |
| generated_reward | -0.078635 | 0.110697 | 1.000000 | -1.813497 | 2.002875 |
| original_env_reward | -1.586450 | 2.399495 | 1.000000 | -100.000000 | 139.587203 |

## Signals
early_failure_or_crash; sparse_proxy:soft_landing_bonus; penalty_dominance:stability_penalty; penalty_dominance:original_env_reward
