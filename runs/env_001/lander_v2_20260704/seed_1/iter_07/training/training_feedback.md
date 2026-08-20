# Training Feedback

## Final-policy outcome
score=-79.469350, len=623.700000, terminated=18/20, truncated=2/20, reward_errors=0
score_range=[-207.681557, 196.947622]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| approach_signal | 284.959215 | 72.4% | 72.4% | 100.0% |
| landing_bonus | 104.751137 | 26.6% | 26.6% | 2.7% |
| stability_penalty | -4.146748 | -1.1% | 1.1% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
