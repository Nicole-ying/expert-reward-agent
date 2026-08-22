# Training Feedback

## Final-policy outcome
score=-464.623326, len=583.050000, terminated=10/20, truncated=10/20, reward_errors=0
score_range=[-912.226149, -2.176882]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_gated | 151.098052 | 69.7% | 80.2% | 71.5% |
| action_penalty | -24.068114 | -11.1% | 11.1% | 100.0% |
| lateral_penalty | -18.919365 | -8.7% | 8.7% | 36.2% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 1/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
