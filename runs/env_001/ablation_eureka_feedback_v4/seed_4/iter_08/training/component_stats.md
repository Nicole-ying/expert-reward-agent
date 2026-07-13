# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.angle_stability | -0.012067 | 0.012067 | 0.999945 | -0.012068 | 0.012068 | -6.835535 | -0.000000 | 1003520 |
| component.approach_progress | 0.014871 | 0.016154 | 0.999403 | 0.014880 | 0.016163 | -0.154900 | 0.185079 | 1003520 |
| component.grounded_quality | 0.181782 | 0.181782 | 0.635055 | 0.286246 | 0.286246 | 0.000000 | 0.300000 | 1003520 |
| component.landing_quality | 0.072684 | 0.072684 | 1.000000 | 0.072684 | 0.072684 | 0.012841 | 0.099947 | 1003520 |
| component.total_reward | 0.191453 | 0.294500 | 1.000000 | 0.191453 | 0.294500 | -7.188772 | 0.399932 | 1003520 |
| component.velocity_penalty | -0.065817 | 0.065817 | 0.320713 | -0.205221 | 0.205221 | -1.255888 | -0.000000 | 1003520 |
| generated_reward | 0.191453 | 0.294500 | 1.000000 | 0.191453 | 0.294500 | -7.188772 | 0.399932 | 1003520 |
| original_env_reward | -0.005287 | 1.295395 | 1.000000 | -0.005287 | 1.295395 | -100.000000 | 115.392674 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| angle_stability | -4.714267 | 4.714267 | -87.722674 | -0.023295 | 2568 |
| approach_progress | 5.801414 | 5.802995 | -1.120960 | 7.094538 | 2568 |
| grounded_quality | 70.795761 | 70.795761 | 0.000000 | 254.607705 | 2568 |
| landing_quality | 28.318040 | 28.318040 | 1.286804 | 91.381448 | 2568 |
| total_reward | 74.506936 | 108.015004 | -114.233060 | 330.739884 | 2568 |
| velocity_penalty | -25.694012 | 25.694012 | -56.058998 | -4.889579 | 2568 |
