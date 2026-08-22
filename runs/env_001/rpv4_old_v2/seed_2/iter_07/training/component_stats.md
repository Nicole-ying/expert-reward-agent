# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.goal_distance_progress | 0.013867 | 0.016075 | 0.997087 | 0.013907 | 0.016122 | -0.134104 | 0.185956 | 1003520 |
| component.landing_contact_success | 1.326594 | 1.326594 | 0.524737 | 2.528112 | 2.528112 | 0.000000 | 4.971514 | 1003520 |
| component.orientation_penalty | -0.014478 | 0.014478 | 0.999766 | -0.014481 | 0.014481 | -7.535320 | -0.000000 | 1003520 |
| component.soft_landing | -0.021806 | 0.021806 | 0.998479 | -0.021839 | 0.021839 | -3.873512 | -0.000000 | 1003520 |
| component.total_reward | 1.304177 | 1.350806 | 1.000000 | 1.304177 | 1.350806 | -8.524674 | 4.971515 | 1003520 |
| generated_reward | 1.304177 | 1.350806 | 1.000000 | 1.304177 | 1.350806 | -8.524674 | 4.971515 | 1003520 |
| original_env_reward | -0.047657 | 1.490656 | 1.000000 | -0.047657 | 1.490656 | -100.000000 | 123.275863 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| goal_distance_progress | 5.259167 | 5.262777 | -1.444159 | 7.096591 | 2641 |
| landing_contact_success | 503.525615 | 503.525615 | 0.000000 | 3937.256509 | 2641 |
| orientation_penalty | -5.498768 | 5.498768 | -173.228520 | -0.010674 | 2641 |
| soft_landing | -8.276456 | 8.276456 | -57.777874 | -1.567975 | 2641 |
| total_reward | 495.009558 | 507.056682 | -172.979256 | 3928.246264 | 2641 |
