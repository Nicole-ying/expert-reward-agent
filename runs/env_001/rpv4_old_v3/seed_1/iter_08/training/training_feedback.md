# Training Feedback

## Final-policy outcome
score=155.496059, len=955.400000, terminated=2/20, truncated=18/20, reward_errors=0
score_range=[114.962962, 252.964397]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact | 138.956703 | 99.3% | 99.3% | 73.7% |
| progress | 0.948563 | 0.7% | 0.7% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
