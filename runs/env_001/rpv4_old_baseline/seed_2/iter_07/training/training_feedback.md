# Training Feedback

## Final-policy outcome
score=-23.830843, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-52.136274, 15.534130]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| state_goodness | 8039.784593 | 99.7% | 99.7% | 100.0% |
| time_penalty | -20.000000 | -0.2% | 0.2% | 100.0% |
| descent_bonus | 1.272571 | 0.0% | 0.0% | 77.5% |
| contact_reward | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
