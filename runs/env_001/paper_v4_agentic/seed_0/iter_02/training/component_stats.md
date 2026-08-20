# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.landing_contact_bonus | 0.283368 | 0.283368 | 0.641961 | 0.441410 | 0.441410 | 0.000000 | 0.599619 | 1003520 |
| component.landing_safety_penalty | 0.003421 | 0.003421 | 1.000000 | 0.003421 | 0.003421 | 0.000000 | 0.451564 | 1003520 |
| component.progress_reward | 0.003086 | 0.003459 | 0.999504 | 0.003088 | 0.003461 | -0.032261 | 0.038492 | 1003520 |
| component.total_reward | 0.283033 | 0.284785 | 1.000000 | 0.283033 | 0.284785 | -0.436644 | 0.601920 | 1003520 |
| component.x_boundary_penalty | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1003520 |
| generated_reward | 0.283033 | 0.284785 | 1.000000 | 0.283033 | 0.284785 | -0.436644 | 0.601920 | 1003520 |
| original_env_reward | -0.020911 | 1.461936 | 1.000000 | -0.020911 | 1.461936 | -100.000000 | 150.603692 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| landing_contact_bonus | 104.832722 | 104.832722 | 0.000000 | 513.064625 | 2708 |
| landing_safety_penalty | 1.266891 | 1.266891 | 0.156331 | 11.809284 | 2708 |
| progress_reward | 1.141901 | 1.142018 | -0.157786 | 1.420258 | 2708 |
| total_reward | 104.707732 | 104.827656 | -3.840695 | 512.795447 | 2708 |
| x_boundary_penalty | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 2708 |
