# Reward Component Training Statistics

- steps_seen: 1120000
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.action_cost | -0.018719 | 0.018719 | 1.000000 | -0.018719 | 0.018719 | -0.040000 | -0.000014 | 1120000 |
| component.contact_transition_reward | 0.049498 | 0.056558 | 0.896358 | 0.055222 | 0.063098 | -0.200000 | 0.187652 | 1120000 |
| component.forward_reward_gated | 0.256205 | 0.256205 | 0.912918 | 0.280644 | 0.280644 | 0.000000 | 1.288426 | 1120000 |
| component.total_reward | 0.286984 | 0.293681 | 1.000000 | 0.286984 | 0.293681 | -0.240000 | 1.444896 | 1120000 |
| generated_reward | 0.286984 | 0.293681 | 1.000000 | 0.286984 | 0.293681 | -0.240000 | 1.444896 | 1120000 |
| original_env_reward | -0.510568 | 0.735097 | 1.000000 | -0.510568 | 0.735097 | -100.000000 | 0.848120 | 1120000 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| action_cost | -3.177365 | 3.177365 | -33.595450 | -0.633739 | 6593 |
| contact_transition_reward | 8.398730 | 8.406532 | -1.516184 | 74.851768 | 6593 |
| forward_reward_gated | 43.464131 | 43.464131 | 0.000000 | 396.426095 | 6593 |
| total_reward | 48.685496 | 48.750282 | -15.035093 | 451.891759 | 6593 |
