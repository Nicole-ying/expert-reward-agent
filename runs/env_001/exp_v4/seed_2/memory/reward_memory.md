# Reward Memory

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
| 9 | landing_shaping + progress_reward + stability_penalty | 67.07 | 67.07 | 0.00 | 841.00 | landing_shaping=2.049 progress_reward=0.206 stability_penalty=-0.079 | no_meaningful_improvement |
| 10 | landing_shaping + progress_reward + stability_penalty | -21.29 | 67.07 | -88.36 | 1000.00 | landing_shaping=2.611 progress_reward=0.306 stability_penalty=-0.084 | unsolved_stagnation_fresh_restart |

## Stable Lessons

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
- stability_penalty weights should be kept low (angle<=0.2, angular<=0.1, speed<=0.15) to avoid dominance
- landing shaping conditions should be relaxed (dist<=1.0, speed<=1.0, angle<=0.5) to achieve nonzero rate above 0.1
- progress_reward coefficient of 50.0 is insufficient to drive learning; increase to 80.0
- landing_shaping with relaxed conditions (dist<=1.0, speed<=1.0, angle<=0.5) leads to high nonzero rate but may cause contact reward hacking
