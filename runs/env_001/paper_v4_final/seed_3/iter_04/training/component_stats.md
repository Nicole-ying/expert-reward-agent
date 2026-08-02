# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.angvel_penalty | -0.000946 | 0.000946 | 0.999866 | -0.000946 | 0.000946 | -2.564496 | -0.000000 | 1003520 |
| component.landing_reward | 1.110796 | 1.110796 | 1.000000 | 1.110796 | 1.110796 | 0.159900 | 1.999323 | 1003520 |
| component.lateral_pos_penalty | -0.025252 | 0.025252 | 0.999995 | -0.025252 | 0.025252 | -0.159605 | -0.000000 | 1003520 |
| component.progress_gated | 0.002909 | 0.012210 | 0.999961 | 0.002909 | 0.012210 | -0.332668 | 0.303270 | 1003520 |
| component.total_reward | 1.087507 | 1.087847 | 1.000000 | 1.087507 | 1.087847 | -2.266135 | 2.061376 | 1003520 |
| generated_reward | 1.087507 | 1.087847 | 1.000000 | 1.087507 | 1.087847 | -2.266135 | 2.061376 | 1003520 |
| original_env_reward | -0.253616 | 1.458840 | 1.000000 | -0.253616 | 1.458840 | -100.000000 | 128.588050 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| angvel_penalty | -0.587199 | 0.587199 | -8.570416 | -0.004668 | 1617 |
| landing_reward | 688.665850 | 688.665850 | 22.573942 | 1493.816591 | 1617 |
| lateral_pos_penalty | -15.654626 | 15.654626 | -114.078739 | -0.000034 | 1617 |
| progress_gated | 1.806656 | 3.380114 | -32.289679 | 12.875823 | 1617 |
| total_reward | 674.230681 | 674.230681 | 21.969190 | 1444.977369 | 1617 |
