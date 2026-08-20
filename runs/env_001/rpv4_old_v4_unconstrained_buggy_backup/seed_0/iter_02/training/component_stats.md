# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.orientation_penalty | -0.008618 | 0.008618 | 0.999995 | -0.008618 | 0.008618 | -3.305949 | -0.000000 | 1003520 |
| component.progress_reward | 0.001086 | 0.002558 | 1.000000 | 0.001086 | 0.002558 | -0.027610 | 0.078326 | 1003520 |
| component.soft_landing_proxy | 3.509147 | 3.509147 | 0.999382 | 3.511316 | 3.511316 | 0.000000 | 4.999994 | 1003520 |
| component.total_reward | 3.441484 | 3.461992 | 1.000000 | 3.441484 | 3.461992 | -3.556710 | 4.999857 | 1003520 |
| component.vel_damping | -0.060131 | 0.060131 | 0.983267 | -0.061154 | 0.061154 | -1.243270 | -0.000000 | 1003520 |
| generated_reward | 3.441484 | 3.461992 | 1.000000 | 3.441484 | 3.461992 | -3.556710 | 4.999857 | 1003520 |
| original_env_reward | -0.207538 | 2.258444 | 1.000000 | -0.207538 | 2.258444 | -100.000000 | 104.300422 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| orientation_penalty | -5.135917 | 5.135917 | -173.484326 | -0.041536 | 1683 |
| progress_reward | 0.645827 | 1.008743 | -10.938180 | 1.412644 | 1683 |
| soft_landing_proxy | 2087.488638 | 2087.488638 | 7.740906 | 4682.777172 | 1683 |
| total_reward | 2047.216934 | 2048.057950 | -75.059822 | 4651.666617 | 1683 |
| vel_damping | -35.781614 | 35.781614 | -79.044012 | -7.086305 | 1683 |
