# Iteration Context

## Recommended Action
**tune** — 本轮外部得分-142.80，远低于目标200。progress_reward均值1.599，信号强但不足以克服stability_penalty和原始环境惩罚。landing_shaping触发率仅1%，稀疏无效。历史中骨架迭代2次，最佳得分158.82（第3轮），但第4轮大幅下降，表明不稳定。建议：增大progress_reward系数至200以上，降低stability_penalty系数至0.05，提高landing_shaping系数至5.0并放宽条件以增加触发率。

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
- score: -142.797815
- episode_length: 88.100000 (mean)
- range: [-206.654021, -3.831011]
- errors: 0

## Component evidence

| component | mean | abs_mean | nonzero_rate | min | max |
|-----------|------|----------|-------------|-----|-----|
| landing_shaping | 0.011850 | 0.011850 | 0.010386 | 0.000000 | 1.981661 |
| progress_reward | 1.598827 | 1.671059 | 0.999991 | -3.217669 | 5.221412 |
| stability_penalty | -0.122471 | 0.122471 | 1.000000 | -0.516735 | -0.000000 |
| total_reward | 1.488206 | 1.571583 | 1.000000 | -3.422873 | 4.793724 |
| generated_reward | 1.488206 | 1.571583 | 1.000000 | -3.422873 | 4.793724 |
| original_env_reward | -1.538719 | 3.915060 | 1.000000 | -100.000000 | 156.650270 |

## Signals
early_failure_or_crash; sparse_proxy:landing_shaping; penalty_dominance:generated_reward; penalty_dominance:original_env_reward
