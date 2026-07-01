# Iteration Context

## Recommended Action
**revert** — 当前得分-108.5远低于历史最佳67.07（差距175.57）。对比best_reward.py，关键差异：progress_reward系数从50降为10，导致学习信号弱；stability_penalty权重过高（angle从0.2→0.5，angular从0.1→0.3，speed从0.15→0.2），导致惩罚主导；soft_landing_bonus条件过严且系数低（2.0 vs 3.0+2.0），激活率极低。建议恢复到best_reward的系数和组件结构（使用landing_shaping替代soft_landing_bonus），仅做小幅调整。

## Agent Memory
| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | progress_reward + soft_landing_bonus + stability_penalty | -108.58 | -108.58 | 0.00 | 72.00 | progress_reward=0.160 soft_landing_bonus=0.012 stability_penalty=-0.342 | new_best |
| 2 | distance_anchor + landing_shaping + progress_reward + stability_penalty | -112.24 | -108.58 | -3.66 | 72.00 | distance_anchor=-0.485 landing_shaping=0.012 progress_reward=0.802 stability_penalty=-0.406 | no_meaningful_improvement |
| 3 | landing_shaping + progress_reward + stability_penalty | -101.17 | -101.17 | 0.00 | 72.00 | landing_shaping=0.013 progress_reward=0.160 stability_penalty=-0.338 | new_best |
| 4 | landing_shaping + progress_reward + stability_penalty | 67.07 | 67.07 | 0.00 | 841.00 | landing_shaping=2.049 progress_reward=0.206 stability_penalty=-0.079 | new_best |
| 5 | landing_shaping + progress_reward + stability_penalty | -113.70 | 67.07 | -180.78 | 72.00 | landing_shaping=0.011 progress_reward=1.291 stability_penalty=-0.238 | no_meaningful_improvement |
| 6 | landing_shaping + progress_reward + stability_penalty | 67.07 | 67.07 | 0.00 | 841.00 | landing_shaping=2.049 progress_reward=0.206 stability_penalty=-0.079 | no_meaningful_improvement |
| 7 | landing_shaping + progress_reward + stability_penalty | -17.49 | 67.07 | -84.57 | 1000.00 | landing_shaping=1.116 progress_reward=0.407 stability_penalty=-0.089 | unsolved_stagnation_fresh_restart |
| 8 | progress_reward + soft_landing_bonus + stability_penalty | -108.50 | 67.07 | -175.57 | 72.00 | progress_reward=0.160 soft_landing_bonus=0.012 stability_penalty=-0.230 | no_meaningful_improvement |

## Expert Cards
## stability_penalty_dominance
- signal: abs(stability_penalty_mean) > abs(progress_reward_mean)
- risk: agent may become overly conservative or afraid to move
- fix: reduce angle/angular-velocity penalty or make it conditional near target

## sparse_completion_proxy
- signal: completion/landing bonus trigger_rate < 1%
- risk: final bonus provides little early learning guidance
- fix: replace hard bonus with smoother landing-quality shaping

## Stable Lessons (from previous iterations)
- Target: mean external score >= 200.
- terminal_success_reward and terminal_failure_penalty blocked (no explicit flag).
- Use external evaluation as fitness signal, not generated_reward alone.
- Contact only inside guarded landing proxy (near target + low speed + stable angle).
- Prefer continuous shaping over hard sparse bonuses.
- distance_anchor can hurt performance when combined with strong progress_reward
- stability_penalty should not dominate total reward; keep mean magnitude below 0.2
- landing_shaping conditions should be relaxed to increase nonzero rate above 0.1
- progress_reward coefficient may need to be increased beyond 50 to achieve meaningful progress
- landing_shaping coefficient should be at least 3.0 to provide meaningful signal
- landing_shaping thresholds should be relaxed (dist<=1.0, speed<=1.0, angle<=0.5) to achieve nonzero rate above 0.1
- progress_reward coefficient of 50 is sufficient; increasing to 80 may cause instability

## Component Evidence
| component | mean | abs_mean | nonzero_rate | min | max |
|-----------|------|----------|-------------|-----|-----|
| progress_reward | 0.160368 | 0.169634 | 0.999992 | -0.408113 | 0.424100 |
| soft_landing_bonus | 0.011723 | 0.011723 | 0.005861 | 0.000000 | 2.000000 |
| stability_penalty | -0.229804 | 0.229804 | 1.000000 | -3.068187 | -0.000147 |
| total_reward | -0.057713 | 0.084969 | 1.000000 | -3.244047 | 2.001660 |
| generated_reward | -0.057713 | 0.084969 | 1.000000 | -3.244047 | 2.001660 |
| original_env_reward | -1.557388 | 2.363114 | 1.000000 | -100.000000 | 129.798128 |
