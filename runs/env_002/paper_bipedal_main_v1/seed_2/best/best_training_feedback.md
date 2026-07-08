# Training Feedback

## Final-policy outcome
score=307.917359, len=1038.650000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[306.582755, 309.758612]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 504.190645 | 82.9% | 83.0% | 100.0% |
| energy_penalty | -74.388368 | -12.2% | 12.2% | 100.0% |
| stability_penalty | -29.165124 | -4.8% | 4.8% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
