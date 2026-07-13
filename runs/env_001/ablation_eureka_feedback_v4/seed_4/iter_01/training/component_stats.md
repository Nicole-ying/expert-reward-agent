# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.angle_stability | -0.040597 | 0.040597 | 0.999970 | -0.040598 | 0.040598 | -14.841504 | -0.000000 | 1003520 |
| component.approach_progress | 0.006994 | 0.008271 | 0.998776 | 0.007003 | 0.008281 | -0.076451 | 0.108563 | 1003520 |
| component.landing_quality | 1.522691 | 1.522691 | 0.363230 | 4.192080 | 4.192080 | 0.000000 | 4.996779 | 1003520 |
| component.total_reward | 0.543652 | 2.484765 | 1.000000 | 0.543652 | 2.484765 | -20.000000 | 4.996768 | 1003520 |
| component.velocity_penalty | -0.945515 | 0.945515 | 0.489807 | -1.930384 | 1.930384 | -20.894392 | -0.000000 | 1003520 |
| generated_reward | 0.543652 | 2.484765 | 1.000000 | 0.543652 | 2.484765 | -20.000000 | 4.996768 | 1003520 |
| original_env_reward | -0.253157 | 1.842639 | 1.000000 | -0.253157 | 1.842639 | -100.000000 | 127.142079 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| angle_stability | -11.879344 | 11.879344 | -251.153043 | -0.044594 | 3427 |
| approach_progress | 2.045392 | 2.109091 | -6.754122 | 2.839946 | 3427 |
| landing_quality | 444.985531 | 444.985531 | 0.000000 | 3815.032765 | 3427 |
| total_reward | 158.487543 | 670.747111 | -1880.420923 | 3660.644508 | 3427 |
| velocity_penalty | -276.687404 | 276.687404 | -1685.612473 | -12.242235 | 3427 |
