# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.goal_distance_progress | 0.013328 | 0.016325 | 0.998379 | 0.013349 | 0.016352 | -0.147454 | 0.175142 | 1003520 |
| component.landing_contact_success | 2.123241 | 2.123241 | 0.445272 | 4.768418 | 4.768418 | 0.000000 | 5.000000 | 1003520 |
| component.orientation_penalty | -0.015734 | 0.015734 | 0.999856 | -0.015736 | 0.015736 | -6.285357 | -0.000000 | 1003520 |
| component.soft_landing | -0.022531 | 0.022531 | 0.999082 | -0.022552 | 0.022552 | -3.857326 | -0.000000 | 1003520 |
| component.total_reward | 2.098304 | 2.147758 | 1.000000 | 2.098304 | 2.147758 | -6.363532 | 5.000236 | 1003520 |
| generated_reward | 2.098304 | 2.147758 | 1.000000 | 2.098304 | 2.147758 | -6.363532 | 5.000236 | 1003520 |
| original_env_reward | -0.067768 | 1.627866 | 1.000000 | -0.067768 | 1.627866 | -100.000000 | 137.965689 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| goal_distance_progress | 5.054910 | 5.060951 | -1.448762 | 7.098830 | 2641 |
| landing_contact_success | 805.978128 | 805.978128 | 0.000000 | 4225.331006 | 2641 |
| orientation_penalty | -5.971123 | 5.971123 | -123.557852 | -0.005531 | 2641 |
| soft_landing | -8.548496 | 8.548496 | -54.713891 | -1.037940 | 2641 |
| total_reward | 796.513419 | 806.942643 | -95.160827 | 4206.366449 | 2641 |
