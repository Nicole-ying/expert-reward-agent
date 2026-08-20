# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.approach_proximity | 0.678782 | 0.678782 | 1.000000 | 0.678782 | 0.678782 | 0.106084 | 0.999920 | 1003520 |
| component.contact_reward | 1.107123 | 1.107123 | 0.369041 | 3.000000 | 3.000000 | 0.000000 | 3.000000 | 1003520 |
| component.soft_landing_velocity | -0.007214 | 0.007214 | 0.998852 | -0.007223 | 0.007223 | -0.960884 | -0.000000 | 1003520 |
| component.total_reward | 1.771079 | 1.772290 | 1.000000 | 1.771079 | 1.772290 | -14.071035 | 3.999730 | 1003520 |
| component.upright_stabilization | -0.007612 | 0.007612 | 0.999988 | -0.007612 | 0.007612 | -14.067599 | -0.000000 | 1003520 |
| generated_reward | 1.771079 | 1.772290 | 1.000000 | 1.771079 | 1.772290 | -14.071035 | 3.999730 | 1003520 |
| original_env_reward | -0.082300 | 1.575826 | 1.000000 | -0.082300 | 1.575826 | -100.000000 | 129.574591 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| approach_proximity | 379.101325 | 379.101325 | 32.109741 | 930.685020 | 1793 |
| contact_reward | 617.754044 | 617.754044 | 0.000000 | 2463.000000 | 1793 |
| soft_landing_velocity | -4.032986 | 4.032986 | -33.854335 | -0.702695 | 1793 |
| total_reward | 988.571048 | 988.571048 | 5.784862 | 3331.487792 | 1793 |
| upright_stabilization | -4.251335 | 4.251335 | -49.079512 | -0.047221 | 1793 |
