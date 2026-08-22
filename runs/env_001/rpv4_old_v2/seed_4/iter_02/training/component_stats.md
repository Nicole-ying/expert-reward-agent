# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.approach_progress | 0.009645 | 0.014679 | 0.997921 | 0.009665 | 0.014710 | -0.181350 | 0.180411 | 1003520 |
| component.contact_completion | 1.537156 | 1.537156 | 0.443621 | 3.465018 | 3.465018 | 0.000000 | 4.999998 | 1003520 |
| component.orientation_penalty | -0.028690 | 0.028690 | 0.999993 | -0.028690 | 0.028690 | -11.622160 | -0.000000 | 1003520 |
| component.soft_landing_speed_penalty | -0.082423 | 0.082423 | 0.995070 | -0.082831 | 0.082831 | -3.659491 | -0.000000 | 1003520 |
| component.total_reward | 1.435688 | 1.598401 | 1.000000 | 1.435688 | 1.598401 | -12.421730 | 4.999969 | 1003520 |
| generated_reward | 1.435688 | 1.598401 | 1.000000 | 1.435688 | 1.598401 | -12.421730 | 4.999969 | 1003520 |
| original_env_reward | -0.223115 | 1.514253 | 1.000000 | -0.223115 | 1.514253 | -100.000000 | 120.401060 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| approach_progress | 3.440141 | 4.276897 | -52.605943 | 7.079142 | 2806 |
| contact_completion | 546.307482 | 546.307482 | 0.000000 | 4153.487215 | 2806 |
| orientation_penalty | -10.230744 | 10.230744 | -183.226232 | -0.050173 | 2806 |
| soft_landing_speed_penalty | -29.402520 | 29.402520 | -194.914774 | -0.002149 | 2806 |
| total_reward | 510.114359 | 543.023570 | -216.065773 | 4101.074476 | 2806 |
