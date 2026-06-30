# Iteration Context

## Recommended Action
**rebuild** — 当前骨架已迭代3轮，最佳得分158.82远低于目标200，且最近两轮得分大幅下降（-142.80, -190.67）。landing_shaping触发率极低（0.2%），无法提供有效学习信号；stability_penalty始终为负且全触发，可能抑制探索。progress_reward均值5.12但外部评分-190.67，说明progress_reward与外部目标不一致。建议重建骨架，移除landing_shaping，大幅提高progress_reward系数（>200），降低stability_penalty系数（<0.1），并考虑加入其他引导信号。

## Agent Memory
| iter | score | best | skeleton_summary | trend |
|------|-------|------|------------------|-------|

## Expert Cards
## early_failure_or_crash
- signal: negative external score and short episode length
- risk: reward does not guide stable control before termination
- fix: add smooth approach/landing signals; avoid relying on sparse terminal-like proxy

## goal_near_oscillation
- signal: distance/progress improves but episode length is long and landing proxy rarely triggers
- risk: agent hovers or oscillates around the goal without completing the task
- fix: add smooth low-speed, low-angle, near-target shaping; avoid pure distance reward

## stability_penalty_dominance
- signal: abs(stability_penalty_mean) > abs(progress_reward_mean)
- risk: agent may become overly conservative or afraid to move
- fix: reduce angle/angular-velocity penalty or make it conditional near target

## Training Evidence
# Training Feedback

## External evaluation
- score: -190.674320
- episode_length: 84.300000 (mean)
- range: [-260.651416, -3.140113]
- errors: 0

## Component evidence

| component | mean | abs_mean | nonzero_rate | min | max |
|-----------|------|----------|-------------|-----|-----|
| landing_shaping | 0.009739 | 0.009739 | 0.002191 | 0.000000 | 9.571960 |
| progress_reward | 5.118749 | 5.338730 | 0.999999 | -9.893801 | 18.333101 |
| stability_penalty | -0.067521 | 0.067521 | 1.000000 | -0.265954 | -0.000000 |
| total_reward | 5.060968 | 5.282389 | 1.000000 | -9.973926 | 18.097342 |
| generated_reward | 5.060968 | 5.282389 | 1.000000 | -9.973926 | 18.097342 |
| original_env_reward | -2.549996 | 4.094505 | 1.000000 | -100.000000 | 144.022630 |

## Signals
early_failure_or_crash; sparse_proxy:landing_shaping; penalty_dominance:generated_reward
