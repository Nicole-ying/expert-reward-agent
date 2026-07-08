# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.approach_reward | 0.087921 | 0.095458 | 0.999159 | 0.087995 | 0.095538 | -0.722883 | 1.086456 | 1003520 |
| component.soft_landing_proxy | 1.672798 | 1.672798 | 0.494129 | 3.385349 | 3.385349 | 0.000000 | 4.995160 | 1003520 |
| component.stability_penalty | -0.002037 | 0.002037 | 1.000000 | -0.002037 | 0.002037 | -0.073039 | -0.000000 | 1003520 |
| component.total_reward | 1.758682 | 1.764257 | 1.000000 | 1.758682 | 1.764257 | -0.637390 | 4.995153 | 1003520 |
| generated_reward | 1.758682 | 1.764257 | 1.000000 | 1.758682 | 1.764257 | -0.637390 | 4.995153 | 1003520 |
| original_env_reward | -0.067513 | 1.657692 | 1.000000 | -0.067513 | 1.657692 | -100.000000 | 128.588050 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| approach_reward | 30.797647 | 30.797647 | 1.473282 | 35.500617 | 2861 |
| soft_landing_proxy | 585.930417 | 585.930417 | 0.000000 | 4106.251571 | 2861 |
| stability_penalty | -0.713652 | 0.713652 | -4.395382 | -0.093673 | 2861 |
| total_reward | 616.014413 | 616.014413 | 0.598625 | 4140.856560 | 2861 |
