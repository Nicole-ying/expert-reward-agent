# Training Feedback

## Final-policy outcome
score=140.269913, len=463.850000, terminated=18/20, truncated=2/20, reward_errors=0
score_range=[-84.830162, 257.016702]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_reward | 2085.000000 | 99.1% | 99.1% | 22.5% |
| progress | 12.121822 | 0.6% | 0.7% | 98.0% |
| ang_penalty | 2.388866 | 0.1% | 0.1% | 9.3% |
| vel_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
