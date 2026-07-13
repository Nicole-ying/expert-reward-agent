# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.angle_stability | -0.017603 | 0.017603 | 1.000000 | -0.017603 | 0.017603 | -7.715027 | -0.000000 | 1003520 |
| component.approach_progress | 0.014565 | 0.017561 | 0.999960 | 0.014566 | 0.017562 | -0.169265 | 0.185172 | 1003520 |
| component.landing_quality | 0.067610 | 0.067610 | 1.000000 | 0.067610 | 0.067610 | 0.012841 | 0.099957 | 1003520 |
| component.total_reward | 0.429630 | 0.552045 | 1.000000 | 0.429630 | 0.552045 | -8.557469 | 10.090930 | 1003520 |
| component.touchdown_success | 0.437438 | 0.437438 | 0.060714 | 7.204865 | 7.204865 | 0.000000 | 9.995376 | 1003520 |
| component.velocity_penalty | -0.072380 | 0.072380 | 0.334053 | -0.216673 | 0.216673 | -1.255888 | -0.000000 | 1003520 |
| generated_reward | 0.429630 | 0.552045 | 1.000000 | 0.429630 | 0.552045 | -8.557469 | 10.090930 | 1003520 |
| original_env_reward | -0.123935 | 4.080443 | 1.000000 | -0.123935 | 4.080443 | -100.000000 | 131.906798 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| angle_stability | -6.688027 | 6.688027 | -104.164788 | -0.012916 | 2639 |
| approach_progress | 5.529051 | 5.530484 | -1.120960 | 7.088606 | 2639 |
| landing_quality | 25.625398 | 25.625398 | 1.287084 | 87.723645 | 2639 |
| total_reward | 162.598176 | 199.085733 | -118.539142 | 854.671528 | 2639 |
| touchdown_success | 165.631523 | 165.631523 | 0.000000 | 785.506184 | 2639 |
| velocity_penalty | -27.499769 | 27.499769 | -71.712117 | -10.606955 | 2639 |
