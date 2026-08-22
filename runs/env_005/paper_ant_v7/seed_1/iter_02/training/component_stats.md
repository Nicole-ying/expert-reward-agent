# Reward Component Training Statistics

- steps_seen: 1001472
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.action_penalty | -0.043113 | 0.043113 | 1.000000 | -0.043113 | 0.043113 | -0.080000 | -0.002000 | 1001472 |
| component.forward_gated | 0.145422 | 0.228312 | 0.343998 | 0.422741 | 0.663701 | -4.037853 | 5.229049 | 1001472 |
| component.lateral_penalty | -0.051356 | 0.051356 | 0.999991 | -0.051356 | 0.051356 | -3.416420 | -0.000000 | 1001472 |
| component.total_reward | 0.050953 | 0.253857 | 1.000000 | 0.050953 | 0.253857 | -5.442130 | 5.172230 | 1001472 |
| generated_reward | 0.050953 | 0.253857 | 1.000000 | 0.050953 | 0.253857 | -5.442130 | 5.172230 | 1001472 |
| original_env_reward | -0.991470 | 1.076262 | 1.000000 | -0.991470 | 1.076262 | -5.660788 | 4.030969 | 1001472 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| action_penalty | -10.576666 | 10.576666 | -52.408813 | -0.285538 | 4081 |
| forward_gated | 35.652156 | 39.239894 | -226.727056 | 486.944615 | 4081 |
| lateral_penalty | -12.598103 | 12.598103 | -134.818121 | -0.045090 | 4081 |
| total_reward | 12.477387 | 33.130831 | -319.359344 | 391.817568 | 4081 |
