# Training Feedback

## Final-policy outcome
score=-113.405732, len=68.400000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-140.041657, -96.181510]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| proximity_stability | 10.757301 | 94.2% | 94.2% | 16.5% |
| fuel_penalty | -0.510000 | -4.5% | 4.5% | 3.7% |
| progress_gated | 0.151162 | 1.3% | 1.3% | 91.9% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
