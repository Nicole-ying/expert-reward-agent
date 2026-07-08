# Training Feedback

## Final-policy outcome
score=304.924958, len=1116.700000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[303.485183, 306.110999]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 504.322799 | 86.1% | 86.1% | 100.0% |
| energy_penalty | -67.630285 | -11.5% | 11.5% | 100.0% |
| stability_penalty | -14.066682 | -2.4% | 2.4% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
