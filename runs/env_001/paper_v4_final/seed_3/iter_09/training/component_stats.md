# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.angle_term_penalty | -0.001560 | 0.001560 | 0.003032 | -0.514446 | 0.514446 | -11.525053 | -0.000000 | 1003520 |
| component.angvel_penalty | -0.007452 | 0.007452 | 0.994996 | -0.007489 | 0.007489 | -2.561706 | -0.000000 | 1003520 |
| component.contact_landing_reward | 0.051276 | 0.051276 | 0.037270 | 1.375815 | 1.375815 | 0.000000 | 4.858905 | 1003520 |
| component.lateral_pos_penalty | -0.003243 | 0.003243 | 0.999968 | -0.003243 | 0.003243 | -0.082341 | -0.000000 | 1003520 |
| component.progress_gated | 0.442214 | 0.471187 | 0.999999 | 0.442215 | 0.471188 | -1.020015 | 1.172846 | 1003520 |
| component.total_reward | 0.481236 | 0.508864 | 1.000000 | 0.481236 | 0.508864 | -11.625946 | 4.716510 | 1003520 |
| generated_reward | 0.481236 | 0.508864 | 1.000000 | 0.481236 | 0.508864 | -11.625946 | 4.716510 | 1003520 |
| original_env_reward | -0.941707 | 2.755711 | 1.000000 | -0.941707 | 2.755711 | -100.000000 | 128.588050 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| angle_term_penalty | -0.114300 | 0.114300 | -95.130908 | 0.000000 | 13696 |
| angvel_penalty | -0.545985 | 0.545985 | -8.570416 | -0.000917 | 13696 |
| contact_landing_reward | 3.757073 | 3.757073 | 0.000000 | 2264.131195 | 13696 |
| lateral_pos_penalty | -0.237571 | 0.237571 | -27.102774 | -0.000001 | 13696 |
| progress_gated | 32.398461 | 32.399553 | -2.937830 | 40.464605 | 13696 |
| total_reward | 35.257677 | 35.438912 | -91.616225 | 2252.289789 | 13696 |
