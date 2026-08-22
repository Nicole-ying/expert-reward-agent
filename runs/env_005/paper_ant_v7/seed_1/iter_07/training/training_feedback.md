# Training Feedback

## Final-policy outcome
score=-381.246922, len=677.500000, terminated=8/20, truncated=12/20, reward_errors=0
score_range=[-983.496188, -18.208524]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress | 707.137389 | 59.5% | 59.5% | 28.2% |
| health_gate | 411.634690 | 34.6% | 34.6% | 61.0% |
| action_penalty | -49.924721 | -4.2% | 4.2% | 100.0% |
| lateral_penalty | -20.025621 | -1.7% | 1.7% | 47.9% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
