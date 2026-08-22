# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.contact_guidance | 0.000493 | 0.000493 | 0.007136 | 0.069054 | 0.069054 | 0.000000 | 0.099480 | 1003520 |
| component.orientation_penalty | -0.004280 | 0.004280 | 1.000000 | -0.004280 | 0.004280 | -1.186124 | -0.000000 | 1003520 |
| component.progress | 0.076027 | 0.080989 | 1.000000 | 0.076027 | 0.080989 | -0.182624 | 0.198658 | 1003520 |
| component.total_reward | 0.065771 | 0.078857 | 1.000000 | 0.065771 | 0.078857 | -1.199674 | 0.172330 | 1003520 |
| component.velocity_penalty | -0.006469 | 0.006469 | 0.997611 | -0.006484 | 0.006484 | -0.070582 | -0.000000 | 1003520 |
| generated_reward | 0.065771 | 0.078857 | 1.000000 | 0.065771 | 0.078857 | -1.199674 | 0.172330 | 1003520 |
| original_env_reward | -0.844951 | 2.825444 | 1.000000 | -0.844951 | 2.825444 | -100.000000 | 132.618997 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| contact_guidance | 0.036291 | 0.036291 | 0.000000 | 48.733343 | 13626 |
| orientation_penalty | -0.315221 | 0.315221 | -7.803790 | -0.005632 | 13626 |
| progress | 5.598699 | 5.598699 | 0.492580 | 7.046810 | 13626 |
| total_reward | 4.843434 | 4.851790 | -4.626476 | 51.355242 | 13626 |
| velocity_penalty | -0.476335 | 0.476335 | -1.303942 | -0.279451 | 13626 |
