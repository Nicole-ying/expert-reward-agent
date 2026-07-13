# Training Feedback

## Final-policy outcome
score=36.532113, len=666.800000, terminated=15/20, truncated=5/20, reward_errors=0
score_range=[-255.000437, 229.080855]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| goal_proximity | -161.169060 | -62.2% | 62.2% | 100.0% |
| contact_reward | 90.207876 | 34.8% | 34.8% | 6.8% |
| velocity_penalty | -6.557522 | -2.5% | 2.5% | 99.8% |
| orientation_penalty | -1.233167 | -0.5% | 0.5% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
