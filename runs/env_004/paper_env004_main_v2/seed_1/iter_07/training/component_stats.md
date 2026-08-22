# Reward Component Training Statistics

- steps_seen: 1000448
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.forward_progress | 1.703832 | 1.703832 | 0.997790 | 1.707605 | 1.707605 | 0.000000 | 5.014320 | 1000448 |
| component.forward_speed_bonus | 0.352507 | 0.352507 | 0.997790 | 0.353288 | 0.353288 | 0.000000 | 2.514341 | 1000448 |
| component.torso_angle_penalty | -0.050276 | 0.050276 | 1.000000 | -0.050276 | 0.050276 | -0.188532 | -0.000000 | 1000448 |
| component.torso_angular_velocity_penalty | -0.121174 | 0.121174 | 1.000000 | -0.121174 | 0.121174 | -1.046646 | -0.000000 | 1000448 |
| component.total_reward | 1.884888 | 1.885853 | 1.000000 | 1.884888 | 1.885853 | -0.986179 | 7.132426 | 1000448 |
| generated_reward | 1.884888 | 1.885853 | 1.000000 | 1.884888 | 1.885853 | -0.986179 | 7.132426 | 1000448 |
| original_env_reward | 2.694498 | 2.694518 | 1.000000 | 2.694498 | 2.694518 | -1.034094 | 5.990506 | 1000448 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| forward_progress | 425.970410 | 425.970410 | 0.005370 | 2120.800655 | 4000 |
| forward_speed_bonus | 88.138912 | 88.138912 | 0.000002 | 487.048560 | 4000 |
| torso_angle_penalty | -12.570926 | 12.570926 | -81.856038 | -0.303529 | 4000 |
| torso_angular_velocity_penalty | -30.295220 | 30.295220 | -187.704579 | -2.430146 | 4000 |
| total_reward | 471.243177 | 471.329848 | -15.523573 | 2382.168597 | 4000 |
