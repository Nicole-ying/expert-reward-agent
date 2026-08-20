# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.fuel_penalty | -0.114726 | 0.114726 | 0.573630 | -0.200000 | 0.200000 | -0.200000 | -0.000000 | 1003520 |
| component.landing | 2.101301 | 2.101301 | 0.956825 | 2.196119 | 2.196119 | 0.000000 | 9.983143 | 1003520 |
| component.progress_gated | 0.449987 | 0.449987 | 0.704885 | 0.638384 | 0.638384 | 0.000000 | 1.996397 | 1003520 |
| component.total_reward | 2.436562 | 2.461929 | 0.993824 | 2.451705 | 2.477229 | -0.200000 | 9.983143 | 1003520 |
| generated_reward | 2.436562 | 2.461929 | 0.993824 | 2.451705 | 2.477229 | -0.200000 | 9.983143 | 1003520 |
| original_env_reward | -0.187761 | 1.766449 | 1.000000 | -0.187761 | 1.766449 | -100.000000 | 116.289347 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| fuel_penalty | -29.172596 | 29.172596 | -179.400000 | -1.000000 | 3941 |
| landing | 534.465982 | 534.465982 | 0.137433 | 7760.828434 | 3941 |
| progress_gated | 114.398452 | 114.398452 | 0.000000 | 562.145823 | 3941 |
| total_reward | 619.691842 | 620.194177 | -54.464515 | 7947.208347 | 3941 |
