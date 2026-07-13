# Training Feedback

## Final-policy outcome
score=-123.863893, len=68.300000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-151.335003, -103.388556]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| approach_improvement | 1.759462 | 39.1% | 39.1% | 1.6% |
| contact_event | 1.400000 | 31.1% | 31.1% | 1.0% |
| progress_delta | 1.114678 | 24.8% | 25.6% | 100.0% |
| fuel_penalty | -0.185000 | -4.1% | 4.1% | 5.4% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
