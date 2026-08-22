# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.attitude_penalty | -0.002303 | 0.002303 | 0.999710 | -0.002304 | 0.002304 | -0.767411 | -0.000000 | 1003520 |
| component.goal_progress | 0.001426 | 0.002806 | 0.999619 | 0.001426 | 0.002807 | -0.056411 | 0.051809 | 1003520 |
| component.leg_contact | 0.212501 | 0.212501 | 0.551899 | 0.385035 | 0.385035 | 0.000000 | 0.400000 | 1003520 |
| component.total_reward | 0.202874 | 0.222092 | 1.000000 | 0.202874 | 0.222092 | -5.880754 | 0.403982 | 1003520 |
| component.velocity_penalty | -0.008749 | 0.008749 | 0.620864 | -0.014092 | 0.014092 | -5.594852 | 0.000000 | 1003520 |
| generated_reward | 0.202874 | 0.222092 | 1.000000 | 0.202874 | 0.222092 | -5.880754 | 0.403982 | 1003520 |
| original_env_reward | -0.329179 | 1.548449 | 1.000000 | -0.329179 | 1.548449 | -100.000000 | 100.000000 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| attitude_penalty | -0.887562 | 0.887562 | -33.884417 | -0.003844 | 2603 |
| goal_progress | 0.547842 | 0.888370 | -9.951306 | 1.420540 | 2603 |
| leg_contact | 81.655551 | 81.655551 | 0.000000 | 345.600000 | 2603 |
| total_reward | 77.946884 | 82.307756 | -82.310930 | 333.351073 | 2603 |
| velocity_penalty | -3.368947 | 3.368947 | -75.304251 | 0.000000 | 2603 |
