# Reward Component Training Statistics

- steps_seen: 1001472
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.forward_progress | 0.557729 | 0.557729 | 0.534391 | 1.043671 | 1.043671 | 0.000000 | 4.232726 | 1001472 |
| component.lateral_drift_hinge | -0.073111 | 0.073111 | 0.411583 | -0.177632 | 0.177632 | -1.804601 | -0.000000 | 1001472 |
| component.roll_pitch_hinge | -0.081185 | 0.081185 | 0.436936 | -0.185805 | 0.185805 | -3.155825 | -0.000000 | 1001472 |
| component.total_reward | 0.403433 | 0.527153 | 0.738584 | 0.546225 | 0.713735 | -3.399637 | 4.109224 | 1001472 |
| generated_reward | 0.403433 | 0.527153 | 0.738584 | 0.546225 | 0.713735 | -3.399637 | 4.109224 | 1001472 |
| original_env_reward | -0.812013 | 1.086602 | 1.000000 | -0.812013 | 1.086602 | -5.985697 | 3.981932 | 1001472 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| forward_progress | 249.082722 | 249.082722 | 0.000000 | 1392.738960 | 2241 |
| lateral_drift_hinge | -32.659082 | 32.659082 | -142.836162 | -0.071765 | 2241 |
| roll_pitch_hinge | -36.267215 | 36.267215 | -186.592863 | -0.038941 | 2241 |
| total_reward | 180.156426 | 205.185933 | -253.838666 | 1193.319953 | 2241 |
