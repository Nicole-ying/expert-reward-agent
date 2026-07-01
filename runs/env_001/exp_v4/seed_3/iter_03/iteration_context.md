# Iteration Context

## Recommended Action
**revert** — 当前得分-112.65低于best的-111.98，差距0.67但方向恶化。对比代码，current将progress_scale从2.0提升到10.0，但stability_penalty系数降低（angle 0.5→0.2, ang_vel 0.3→0.1, speed 0.2→0.1），landing_bonus条件放宽（dist<0.8, speed<0.5, angle<0.3）且奖励从1.0提升到2.0。这些修改导致progress_reward均值从0.032提升到0.161，但stability_penalty均值从-0.243改善到-0.120，然而score下降，说明过度放宽约束和增加progress权重可能引发震荡或未改善着陆。建议恢复到best的系数，仅做小幅调整。

## Agent Memory
| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | energy_penalty + landing_bonus + progress_reward + stability_penalty | -111.98 | -111.98 | 0.00 | 70.60 | energy_penalty=-0.012 landing_bonus=0.005 progress_reward=0.032 stability_penalty=-0.243 | new_best |
| 2 | energy_penalty + landing_bonus + progress_reward + stability_penalty | -112.65 | -111.98 | -0.67 | 70.60 | energy_penalty=-0.010 landing_bonus=0.014 progress_reward=0.161 stability_penalty=-0.120 | no_meaningful_improvement |

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

## Component Evidence
| component | mean | abs_mean | nonzero_rate | min | max |
|-----------|------|----------|-------------|-----|-----|
| energy_penalty | -0.010358 | 0.010358 | 0.103577 | -0.100000 | -0.000000 |
| landing_bonus | 0.013845 | 0.013845 | 0.006923 | 0.000000 | 2.000000 |
| progress_reward | 0.161149 | 0.170547 | 0.999996 | -0.408164 | 0.421506 |
| stability_penalty | -0.119596 | 0.119596 | 1.000000 | -1.508314 | -0.000000 |
| total_reward | 0.045040 | 0.108360 | 1.000000 | -1.771398 | 2.035966 |
| generated_reward | 0.045040 | 0.108360 | 1.000000 | -1.771398 | 2.035966 |
| original_env_reward | -1.608536 | 2.315101 | 1.000000 | -100.000000 | 128.588050 |
