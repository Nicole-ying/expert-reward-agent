# Training Feedback

## Final-policy outcome
score=146.039486, len=963.850000, terminated=1/20, truncated=19/20, reward_errors=0
score_range=[97.983492, 271.953181]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress_reward | 21.950159 | 90.4% | 93.3% | 99.9% |
| attitude_penalty | -1.629122 | -6.7% | 6.7% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
