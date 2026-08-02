# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.distance_progress | 0.002127 | 0.002391 | 0.999576 | 0.002128 | 0.002392 | -0.029199 | 0.033768 | 1003520 |
| component.orientation_penalty | -0.002178 | 0.002178 | 0.999780 | -0.002179 | 0.002179 | -1.583273 | -0.000000 | 1003520 |
| component.soft_landing_bonus | 0.278231 | 0.278231 | 0.603479 | 0.461046 | 0.461046 | 0.000000 | 0.499999 | 1003520 |
| component.total_reward | 0.271737 | 0.284114 | 1.000000 | 0.271737 | 0.284114 | -1.863063 | 0.500003 | 1003520 |
| component.velocity_damping | -0.006443 | 0.006443 | 0.999809 | -0.006445 | 0.006445 | -0.524146 | -0.000000 | 1003520 |
| generated_reward | 0.271737 | 0.284114 | 1.000000 | 0.271737 | 0.284114 | -1.863063 | 0.500003 | 1003520 |
| original_env_reward | 0.044427 | 1.273072 | 1.000000 | 0.044427 | 1.273072 | -100.000000 | 114.015943 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| distance_progress | 1.137960 | 1.143534 | -0.609247 | 1.419158 | 1871 |
| orientation_penalty | -1.167987 | 1.167987 | -21.648064 | -0.005949 | 1871 |
| soft_landing_bonus | 148.878575 | 148.878575 | 0.000000 | 422.694491 | 1871 |
| total_reward | 145.397259 | 150.140861 | -34.066698 | 421.922061 | 1871 |
| velocity_damping | -3.451290 | 3.451290 | -18.028368 | -1.270676 | 1871 |
