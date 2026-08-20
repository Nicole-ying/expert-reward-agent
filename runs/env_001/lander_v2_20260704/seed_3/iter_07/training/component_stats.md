# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.approach_reward | 0.002733 | 0.003209 | 0.998318 | 0.002738 | 0.003214 | -0.026935 | 0.035955 | 1003520 |
| component.soft_landing_proxy | 1.691058 | 1.691058 | 0.585409 | 2.888676 | 2.888676 | 0.000000 | 4.992935 | 1003520 |
| component.stability_penalty | -0.001711 | 0.001711 | 1.000000 | -0.001711 | 0.001711 | -0.058941 | -0.000000 | 1003520 |
| component.total_reward | 1.692081 | 1.692652 | 1.000000 | 1.692081 | 1.692652 | -0.078727 | 4.992934 | 1003520 |
| generated_reward | 1.692081 | 1.692652 | 1.000000 | 1.692081 | 1.692652 | -0.078727 | 4.992934 | 1003520 |
| original_env_reward | 0.026531 | 1.645603 | 1.000000 | 0.026531 | 1.645603 | -100.000000 | 128.588050 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| approach_reward | 1.144979 | 1.145150 | -0.179680 | 1.415400 | 2392 |
| soft_landing_proxy | 707.984731 | 707.984731 | 0.000000 | 4126.140863 | 2392 |
| stability_penalty | -0.716551 | 0.716551 | -6.318081 | -0.097274 | 2392 |
| total_reward | 708.413159 | 708.434636 | -1.141833 | 4126.843895 | 2392 |
