# Training Feedback

## Final-policy outcome
score=114.499268, len=907.150000, terminated=4/20, truncated=16/20, reward_errors=0
score_range=[38.473905, 267.678322]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| safe_contact_bonus | 521.688581 | 95.6% | 95.6% | 8.3% |
| fuel_penalty | -13.953000 | -2.6% | 2.6% | 76.9% |
| stability_penalty | -7.026677 | -1.3% | 1.3% | 100.0% |
| progress_reward | 2.672819 | 0.5% | 0.6% | 99.7% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
