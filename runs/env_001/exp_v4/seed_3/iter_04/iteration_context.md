# Iteration Context

## Recommended Action
**tune** — 当前骨架已运行3轮，得分稳定在-110左右，无显著改善。best_reward.py与previous_reward.py几乎相同（仅landing_bonus从1.0提升到1.5），但best得分-111.98略低于当前-110.51，因此无需revert。主要问题是stability_penalty系数过高（0.5,0.3,0.2）导致其主导总奖励，建议降低系数（如0.2,0.1,0.1）以鼓励探索；同时progress_reward系数2.0可能过低，可尝试提升至5.0或10.0以增强学习信号；landing_bonus条件可适当放宽（如dist<0.8, speed<0.5, angle<0.3）以提高触发率。

## Agent Memory
| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | energy_penalty + landing_bonus + progress_reward + stability_penalty | -111.98 | -111.98 | 0.00 | 70.60 | energy_penalty=-0.012 landing_bonus=0.005 progress_reward=0.032 stability_penalty=-0.243 | new_best |
| 2 | energy_penalty + landing_bonus + progress_reward + stability_penalty | -112.65 | -111.98 | -0.67 | 70.60 | energy_penalty=-0.010 landing_bonus=0.014 progress_reward=0.161 stability_penalty=-0.120 | no_meaningful_improvement |
| 3 | energy_penalty + landing_bonus + progress_reward + stability_penalty | -110.51 | -111.98 | 1.47 | 70.60 | energy_penalty=-0.013 landing_bonus=0.008 progress_reward=0.032 stability_penalty=-0.242 | no_meaningful_improvement |

## Expert Cards
## goal_near_oscillation
- signal: distance/progress improves but episode length is long and landing proxy rarely triggers
- risk: agent hovers or oscillates around the goal without completing the task
- fix: add smooth low-speed, low-angle, near-target shaping; avoid pure distance reward

## stability_penalty_dominance
- signal: abs(stability_penalty_mean) > abs(progress_reward_mean)
- risk: agent may become overly conservative or afraid to move
- fix: reduce angle/angular-velocity penalty or make it conditional near target

## Stable Lessons (from previous iterations)
- Target: mean external score >= 200.
- terminal_success_reward and terminal_failure_penalty blocked (no explicit flag).
- Use external evaluation as fitness signal, not generated_reward alone.
- Contact only inside guarded landing proxy (near target + low speed + stable angle).
- Prefer continuous shaping over hard sparse bonuses.
- stability_penalty coefficients should be tuned to avoid dominating the total reward
- progress_reward coefficient increase from 2.0 to 10.0 did not improve score, likely due to goal_near_oscillation
- stability_penalty coefficients reduction from (0.5,0.3,0.2) to (0.2,0.1,0.1) reduced penalty but did not improve overall performance
- landing_bonus condition relaxation (dist<0.8, speed<0.5, angle<0.3) and reward increase to 2.0 did not increase landing rate (nonzero rate still very low)

## Component Evidence
| component | mean | abs_mean | nonzero_rate | min | max |
|-----------|------|----------|-------------|-----|-----|
| energy_penalty | -0.012977 | 0.012977 | 0.129772 | -0.100000 | -0.000000 |
| landing_bonus | 0.008281 | 0.008281 | 0.005521 | 0.000000 | 1.500000 |
| progress_reward | 0.032392 | 0.034250 | 0.999992 | -0.080574 | 0.084193 |
| stability_penalty | -0.241722 | 0.241722 | 1.000000 | -3.432804 | -0.000000 |
| total_reward | -0.214026 | 0.229844 | 1.000000 | -3.572376 | 1.499441 |
| generated_reward | -0.214026 | 0.229844 | 1.000000 | -3.572376 | 1.499441 |
| original_env_reward | -1.581722 | 2.299603 | 1.000000 | -100.000000 | 128.588050 |
