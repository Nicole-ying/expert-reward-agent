# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.approach_reward | 0.004091 | 0.004531 | 0.999597 | 0.004092 | 0.004533 | -0.030237 | 0.037003 | 1003520 |
| component.engine_penalty | -0.003580 | 0.003580 | 0.715942 | -0.005000 | 0.005000 | -0.005000 | -0.000000 | 1003520 |
| component.soft_landing_proxy | 1.660945 | 1.660945 | 0.589338 | 2.818325 | 2.818325 | 0.000000 | 4.992863 | 1003520 |
| component.stability_penalty | -0.001502 | 0.001502 | 1.000000 | -0.001502 | 0.001502 | -0.058941 | -0.000000 | 1003520 |
| component.total_reward | 1.659954 | 1.661353 | 1.000000 | 1.659954 | 1.661353 | -0.083727 | 4.992863 | 1003520 |
| generated_reward | 1.659954 | 1.661353 | 1.000000 | 1.659954 | 1.661353 | -0.083727 | 4.992863 | 1003520 |
| original_env_reward | -0.065179 | 1.520727 | 1.000000 | -0.065179 | 1.520727 | -100.000000 | 128.588050 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| approach_reward | 1.110862 | 1.110883 | -0.038744 | 1.418069 | 3691 |
| engine_penalty | -0.971857 | 0.971857 | -4.415000 | -0.165000 | 3691 |
| soft_landing_proxy | 450.671168 | 450.671168 | 0.000000 | 4137.008744 | 3691 |
| stability_penalty | -0.407869 | 0.407869 | -4.453870 | -0.080646 | 3691 |
| total_reward | 450.402304 | 450.439833 | -2.983632 | 4134.923561 | 3691 |
