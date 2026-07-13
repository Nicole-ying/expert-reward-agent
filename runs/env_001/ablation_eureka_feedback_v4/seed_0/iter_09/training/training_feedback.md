# Training Feedback

## Final-policy outcome
score=216.335063, len=513.600000, terminated=15/20, truncated=5/20, reward_errors=0
score_range=[82.565113, 274.172355]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| stability | 505.993467 | 62.5% | 62.5% | 100.0% |
| contact | 279.205479 | 34.5% | 34.5% | 45.7% |
| progress | 12.218263 | 1.5% | 1.5% | 72.7% |
| engine_penalty | -11.746500 | -1.5% | 1.5% | 76.2% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
