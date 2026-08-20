# Reward Component Training Statistics

- steps_seen: 1001472
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.orientation_penalty | -0.004040 | 0.004040 | 1.000000 | -0.004040 | 0.004040 | -5.206807 | -0.000000 | 1001472 |
| component.proximity_reward | 0.029232 | 0.031442 | 0.999996 | 0.029232 | 0.031442 | -0.081471 | 0.083483 | 1001472 |
| component.settlement_bonus | 0.425562 | 0.458736 | 0.043024 | 9.891341 | 10.662412 | -47.681948 | 121.979133 | 1001472 |
| component.speed_penalty_gated | -0.154587 | 0.154587 | 0.999999 | -0.154587 | 0.154587 | -1.678752 | -0.000000 | 1001472 |
| component.total_reward | 0.015495 | 0.275969 | 1.000000 | 0.015495 | 0.275969 | -20.000000 | 20.000000 | 1001472 |
| generated_reward | 0.015495 | 0.275969 | 1.000000 | 0.015495 | 0.275969 | -20.000000 | 20.000000 | 1001472 |
| original_env_reward | -1.186406 | 2.673183 | 1.000000 | -1.186406 | 2.673183 | -100.000000 | 140.711243 | 1001472 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| orientation_penalty | -0.305686 | 0.305686 | -37.939311 | -0.000875 | 13237 |
| proximity_reward | 2.211495 | 2.215055 | -3.162023 | 2.827643 | 13237 |
| settlement_bonus | 32.196738 | 32.196738 | -0.000000 | 124.152591 | 13237 |
| speed_penalty_gated | -11.695368 | 11.695368 | -29.353180 | -4.115638 | 13237 |
| total_reward | 1.172428 | 11.179572 | -60.399073 | 88.190762 | 13237 |
