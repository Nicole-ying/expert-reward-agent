# Training Feedback

## Final-policy outcome
score=-109.491590, len=68.450000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-124.079149, -91.955868]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing | 5.400996 | 51.4% | 51.4% | 97.8% |
| progress_delta | 4.490279 | 42.7% | 42.7% | 92.0% |
| fuel_penalty | -0.620000 | -5.9% | 5.9% | 4.5% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
