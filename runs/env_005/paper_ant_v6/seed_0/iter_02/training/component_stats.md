# Reward Component Training Statistics

- steps_seen: 1001472
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.forward_reward | 0.551648 | 0.551648 | 0.755933 | 0.729757 | 0.729757 | 0.000000 | 4.347964 | 1001472 |
| component.height_penalty | -0.000586 | 0.000586 | 0.001691 | -0.346703 | 0.346703 | -2.250876 | -0.000000 | 1001472 |
| component.lateral_penalty | -0.093491 | 0.093491 | 0.999924 | -0.093498 | 0.093498 | -7.597813 | -0.000000 | 1001472 |
| component.total_reward | 0.457571 | 0.554382 | 0.999960 | 0.457589 | 0.554404 | -7.597813 | 4.345902 | 1001472 |
| generated_reward | 0.457571 | 0.554382 | 0.999960 | 0.457589 | 0.554404 | -7.597813 | 4.345902 | 1001472 |
| original_env_reward | -0.843534 | 1.102980 | 1.000000 | -0.843534 | 1.102980 | -5.875333 | 4.111433 | 1001472 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| forward_reward | 228.695416 | 228.695416 | 0.000000 | 1302.346463 | 2414 |
| height_penalty | -0.243152 | 0.243152 | -2.250876 | 0.000000 | 2414 |
| lateral_penalty | -38.770732 | 38.770732 | -253.387563 | -0.363951 | 2414 |
| total_reward | 189.681533 | 210.551725 | -243.363753 | 1234.582704 | 2414 |
