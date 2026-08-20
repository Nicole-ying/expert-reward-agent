# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.attitude_penalty | -0.000784 | 0.000784 | 0.999751 | -0.000784 | 0.000784 | -1.181762 | -0.000000 | 1003520 |
| component.proximity | 0.733106 | 0.733106 | 1.000000 | 0.733106 | 0.733106 | 0.075866 | 0.999940 | 1003520 |
| component.soft_landing_bonus | 0.448688 | 0.448688 | 0.509023 | 0.881469 | 0.881469 | 0.000000 | 0.999998 | 1003520 |
| component.total_reward | 1.179711 | 1.179713 | 1.000000 | 1.179711 | 1.179713 | -0.556665 | 1.999075 | 1003520 |
| component.velocity_penalty | -0.001299 | 0.001299 | 0.996820 | -0.001303 | 0.001303 | -0.218705 | -0.000000 | 1003520 |
| generated_reward | 1.179711 | 1.179713 | 1.000000 | 1.179711 | 1.179713 | -0.556665 | 1.999075 | 1003520 |
| original_env_reward | -0.066706 | 1.554277 | 1.000000 | -0.066706 | 1.554277 | -100.000000 | 125.453575 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| attitude_penalty | -0.461010 | 0.461010 | -7.562551 | -0.004349 | 1705 |
| proximity | 430.189478 | 430.189478 | 32.286057 | 945.544824 | 1705 |
| soft_landing_bonus | 263.079918 | 263.079918 | 0.000000 | 849.516585 | 1705 |
| total_reward | 692.045110 | 692.045110 | 31.095976 | 1789.334384 | 1705 |
| velocity_penalty | -0.763278 | 0.763278 | -11.184487 | -0.146009 | 1705 |
