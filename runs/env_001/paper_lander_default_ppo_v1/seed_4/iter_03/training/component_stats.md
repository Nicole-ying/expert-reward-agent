# Reward Component Training Statistics

- steps_seen: 1001472
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.orientation_penalty | -0.004300 | 0.004300 | 0.999901 | -0.004301 | 0.004301 | -4.854291 | -0.000000 | 1001472 |
| component.proximity_reward | 0.004943 | 0.005851 | 0.988611 | 0.005000 | 0.005918 | -0.091018 | 0.093965 | 1001472 |
| component.settlement_bonus | 0.933977 | 0.933977 | 0.640941 | 1.457197 | 1.457197 | 0.000000 | 1.998853 | 1001472 |
| component.speed_penalty_gated | -0.015789 | 0.015789 | 0.993021 | -0.015900 | 0.015900 | -1.633656 | -0.000000 | 1001472 |
| component.total_reward | 0.918830 | 0.946868 | 1.000000 | 0.918830 | 0.946868 | -5.761648 | 1.998861 | 1001472 |
| generated_reward | 0.918830 | 0.946868 | 1.000000 | 0.918830 | 0.946868 | -5.761648 | 1.998861 | 1001472 |
| original_env_reward | 0.091473 | 1.110859 | 1.000000 | 0.091473 | 1.110859 | -100.000000 | 102.195715 | 1001472 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| orientation_penalty | -1.964815 | 1.964815 | -243.193697 | -0.025463 | 2191 |
| proximity_reward | 2.257973 | 2.393096 | -11.405716 | 2.840216 | 2191 |
| settlement_bonus | 426.632597 | 426.632597 | 0.000000 | 1750.376793 | 2191 |
| speed_penalty_gated | -7.212669 | 7.212669 | -41.600725 | -1.589603 | 2191 |
| total_reward | 419.713085 | 428.108627 | -271.987223 | 1746.711383 | 2191 |
