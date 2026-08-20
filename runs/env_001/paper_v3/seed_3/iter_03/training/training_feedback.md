# Training Feedback

## Final-policy outcome
score=-25.511089, len=971.850000, terminated=1/20, truncated=19/20, reward_errors=0
score_range=[-59.802101, 25.998083]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_reward | 31.350000 | 80.6% | 80.6% | 1.1% |
| approach_and_soft_landing | 3.975440 | 10.2% | 17.6% | 100.0% |
| upright_stabilization | -0.689845 | -1.8% | 1.8% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
