# Training Feedback

## Final-policy outcome
score=-0.235020, len=483.850000, terminated=19/20, truncated=1/20, reward_errors=0
score_range=[-204.442924, 202.102233]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_reward | 566.416414 | 98.3% | 98.3% | 7.3% |
| progress | 4.902541 | 0.9% | 1.7% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
