# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.angvel_penalty | -0.009217 | 0.009217 | 0.994408 | -0.009269 | 0.009269 | -3.376828 | -0.000000 | 1003520 |
| component.contact_landing_reward | 0.048460 | 0.048460 | 0.034193 | 1.417262 | 1.417262 | 0.000000 | 4.744866 | 1003520 |
| component.lateral_pos_penalty | -0.003043 | 0.003043 | 0.999970 | -0.003043 | 0.003043 | -0.081995 | -0.000000 | 1003520 |
| component.progress_gated | 0.459144 | 0.486528 | 0.999990 | 0.459149 | 0.486533 | -1.125710 | 1.203065 | 1003520 |
| component.total_reward | 0.495344 | 0.522844 | 1.000000 | 0.495344 | 0.522844 | -3.538095 | 4.716510 | 1003520 |
| generated_reward | 0.495344 | 0.522844 | 1.000000 | 0.495344 | 0.522844 | -3.538095 | 4.716510 | 1003520 |
| original_env_reward | -1.266711 | 2.573154 | 1.000000 | -1.266711 | 2.573154 | -100.000000 | 128.588050 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| angvel_penalty | -0.656098 | 0.656098 | -8.570416 | -0.001542 | 14098 |
| contact_landing_reward | 3.449462 | 3.449462 | 0.000000 | 2264.364324 | 14098 |
| lateral_pos_penalty | -0.216570 | 0.216570 | -27.102818 | -0.000000 | 14098 |
| progress_gated | 32.680950 | 32.682832 | -5.105660 | 40.429197 | 14098 |
| total_reward | 35.257744 | 35.264995 | -7.352467 | 2252.565626 | 14098 |
