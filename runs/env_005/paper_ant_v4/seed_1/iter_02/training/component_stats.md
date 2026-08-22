# Reward Component Training Statistics

- steps_seen: 1015808
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.forward_progress | 1.056757 | 5.618784 | 1.000000 | 1.056757 | 5.618784 | -16.219350 | 16.593289 | 1015808 |
| component.height_penalty | -0.374456 | 0.374456 | 0.514315 | -0.728068 | 0.728068 | -1.999964 | 0.000000 | 1015808 |
| component.lateral_drift_penalty | -0.043700 | 0.043700 | 0.999994 | -0.043700 | 0.043700 | -2.021438 | -0.000000 | 1015808 |
| component.total_reward | 0.493099 | 5.796823 | 1.000000 | 0.493099 | 5.796823 | -16.615731 | 17.432732 | 1015808 |
| component.upright_stability | -0.145503 | 0.938782 | 1.000000 | -0.145503 | 0.938782 | -1.609042 | 0.999982 | 1015808 |
| generated_reward | 0.493099 | 5.796823 | 1.000000 | 0.493099 | 5.796823 | -16.615731 | 17.432732 | 1015808 |
| original_env_reward | -0.845906 | 1.019063 | 1.000000 | -0.845906 | 1.019063 | -5.982240 | 4.332202 | 1015808 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| forward_progress | 167.121709 | 241.079904 | -1010.222707 | 3048.360930 | 6363 |
| height_penalty | -59.691475 | 59.691475 | -696.102590 | 0.000000 | 6363 |
| lateral_drift_penalty | -6.956129 | 6.956129 | -78.210873 | -0.018394 | 6363 |
| total_reward | 77.081239 | 350.475123 | -2228.856586 | 3552.676919 | 6363 |
| upright_stability | -23.392866 | 135.028550 | -938.111161 | 946.102274 | 6363 |
