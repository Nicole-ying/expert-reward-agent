# Reward Component Training Statistics

- steps_seen: 2000000
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.forward_progress | 0.246542 | 0.255084 | 0.999918 | 0.246562 | 0.255105 | -0.599423 | 0.851125 | 2000000 |
| component.ground_penalty | -0.001769 | 0.001769 | 0.005895 | -0.300000 | 0.300000 | -0.300000 | 0.000000 | 2000000 |
| component.stability_angle_penalty | -0.011852 | 0.011852 | 0.091971 | -0.128871 | 0.128871 | -34.559944 | -0.000000 | 2000000 |
| component.total_reward | 0.232991 | 0.257757 | 0.999918 | 0.233010 | 0.257778 | -20.000000 | 0.851125 | 2000000 |
| generated_reward | 0.232991 | 0.257757 | 0.999918 | 0.233010 | 0.257778 | -20.000000 | 0.851125 | 2000000 |
| original_env_reward | -0.392091 | 0.654157 | 1.000000 | -0.392091 | 0.654157 | -100.000000 | 0.941059 | 2000000 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| forward_progress | 49.800932 | 49.976203 | -15.223886 | 414.229288 | 9890 |
| ground_penalty | -0.357634 | 0.357634 | -13.200000 | 0.000000 | 9890 |
| stability_angle_penalty | -2.395606 | 2.395606 | -569.141842 | 0.000000 | 9890 |
| total_reward | 47.061809 | 49.195196 | -481.958649 | 398.679815 | 9890 |
