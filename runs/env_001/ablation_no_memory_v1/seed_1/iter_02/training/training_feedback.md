# Training Feedback

## Final-policy outcome
score=-109.400689, len=68.500000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-123.242094, -78.933292]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| stability_penalty | -7.544432 | -66.3% | 66.3% | 100.0% |
| progress_delta | 3.361815 | 29.6% | 30.6% | 100.0% |
| soft_landing_reward | 0.346006 | 3.0% | 3.0% | 96.1% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
