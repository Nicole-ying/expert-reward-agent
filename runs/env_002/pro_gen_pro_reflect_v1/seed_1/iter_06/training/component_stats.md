# Reward Component Training Statistics

- steps_seen: 1015808
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.action_efficiency | -0.035506 | 0.035506 | 1.000000 | -0.035506 | 0.035506 | -0.080000 | -0.000019 | 1015808 |
| component.stability_bonus | 0.134914 | 0.135742 | 0.999999 | 0.134915 | 0.135742 | -0.109418 | 0.402524 | 1015808 |
| component.total_reward | 0.411965 | 0.423391 | 1.000000 | 0.411965 | 0.423391 | -0.388558 | 1.176183 | 1015808 |
| component.velocity_gated | 0.312556 | 0.315035 | 0.990865 | 0.315437 | 0.317940 | -0.285813 | 0.857110 | 1015808 |
| generated_reward | 0.411965 | 0.423391 | 1.000000 | 0.411965 | 0.423391 | -0.388558 | 1.176183 | 1015808 |
| original_env_reward | -0.446812 | 0.798668 | 1.000000 | -0.446812 | 0.798668 | -100.000000 | 0.801526 | 1015808 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| action_efficiency | -5.846659 | 5.846659 | -67.354176 | -1.385883 | 6153 |
| stability_bonus | 22.176706 | 22.184747 | -2.846155 | 246.427326 | 6153 |
| total_reward | 67.723764 | 68.412331 | -84.699323 | 717.055112 | 6153 |
| velocity_gated | 51.393717 | 51.445968 | -16.465080 | 504.960279 | 6153 |
