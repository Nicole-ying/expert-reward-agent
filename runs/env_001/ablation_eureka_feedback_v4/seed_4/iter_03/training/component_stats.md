# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.angle_stability | -0.031530 | 0.031530 | 0.999976 | -0.031531 | 0.031531 | -9.953478 | -0.000000 | 1003520 |
| component.approach_progress | 0.005971 | 0.007075 | 0.999803 | 0.005972 | 0.007076 | -0.060629 | 0.096212 | 1003520 |
| component.landing_quality | 0.061686 | 0.061686 | 1.000000 | 0.061686 | 0.061686 | 0.003410 | 0.099983 | 1003520 |
| component.total_reward | -0.698310 | 0.802900 | 1.000000 | -0.698310 | 0.802900 | -20.000000 | 0.099969 | 1003520 |
| component.velocity_penalty | -0.734438 | 0.734438 | 0.302903 | -2.424666 | 2.424666 | -18.099901 | -0.000000 | 1003520 |
| generated_reward | -0.698310 | 0.802900 | 1.000000 | -0.698310 | 0.802900 | -20.000000 | 0.099969 | 1003520 |
| original_env_reward | -0.225599 | 2.499879 | 1.000000 | -0.225599 | 2.499879 | -100.000000 | 135.177011 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| angle_stability | -11.086197 | 11.086197 | -194.363799 | -0.031831 | 2853 |
| approach_progress | 2.097544 | 2.183608 | -11.335945 | 2.836346 | 2853 |
| landing_quality | 21.675239 | 21.675239 | 1.385324 | 85.608232 | 2853 |
| total_reward | -245.561214 | 263.286780 | -1527.281989 | 80.557231 | 2853 |
| velocity_penalty | -258.248174 | 258.248174 | -1458.013377 | 0.000000 | 2853 |
