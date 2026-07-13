# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.fuel_cost | -0.012925 | 0.012925 | 0.086164 | -0.150000 | 0.150000 | -0.150000 | -0.000000 | 1003520 |
| component.proximity_reward | 0.252731 | 0.269250 | 0.999501 | 0.252857 | 0.269384 | -0.385116 | 0.483912 | 1003520 |
| component.soft_landing_bonus | 0.022381 | 0.022381 | 0.017231 | 1.298856 | 1.298856 | 0.000000 | 1.998989 | 1003520 |
| component.total_reward | 0.262187 | 0.287270 | 0.999858 | 0.262224 | 0.287311 | -0.485298 | 1.998949 | 1003520 |
| component.velocity_penalty | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | -0.000000 | -0.000000 | 1003520 |
| generated_reward | 0.262187 | 0.287270 | 0.999858 | 0.262224 | 0.287311 | -0.485298 | 1.998949 | 1003520 |
| original_env_reward | -1.588244 | 2.290144 | 1.000000 | -1.588244 | 2.290144 | -100.000000 | 116.874017 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| fuel_cost | -0.905815 | 0.905815 | -114.600000 | 0.000000 | 14316 |
| proximity_reward | 17.713017 | 17.725312 | -7.636586 | 19.383856 | 14316 |
| soft_landing_bonus | 1.568862 | 1.568862 | 0.000000 | 889.806847 | 14316 |
| total_reward | 18.376063 | 18.515955 | -21.286587 | 785.144047 | 14316 |
| velocity_penalty | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 14316 |
