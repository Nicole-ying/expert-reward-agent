# Training Feedback

## Final-policy outcome
score=-353.939790, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-1388.713726, -20.188342]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| gated_forward | 2042.572918 | 57.8% | 58.4% | 90.3% |
| height_gate | 931.002856 | 26.3% | 26.3% | 100.0% |
| joint_vel_penalty | -376.686212 | -10.7% | 10.7% | 90.1% |
| upright_reward | 72.126853 | 2.0% | 2.7% | 100.0% |
| lateral_penalty | -66.388156 | -1.9% | 1.9% | 90.1% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
