# Training Feedback

## Final-policy outcome
score=13.548498, len=740.650000, terminated=9/20, truncated=11/20, reward_errors=0
score_range=[-615.841894, 354.626578]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_gated | 823.134702 | 88.5% | 90.2% | 79.3% |
| lateral_penalty | -59.744231 | -6.4% | 6.4% | 77.7% |
| action_penalty | -31.733037 | -3.4% | 3.4% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
