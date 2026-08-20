# Training Feedback

## Final-policy outcome
score=-54.722843, len=162.350000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-780.973055, 8.898390]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_bonus | 174.654002 | 92.7% | 92.7% | 33.7% |
| progress | 4.250309 | 2.3% | 4.4% | 100.0% |
| fuel_penalty | -4.930000 | -2.6% | 2.6% | 60.7% |
| orientation_penalty | -0.542035 | -0.3% | 0.3% | 17.3% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
