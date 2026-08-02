# Training Feedback

## Final-policy outcome
score=144.813937, len=960.100000, terminated=1/20, truncated=19/20, reward_errors=0
score_range=[73.665443, 183.494761]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_reward | 140.110002 | 98.7% | 98.7% | 77.2% |
| progress_delta | 1.372766 | 1.0% | 1.0% | 100.0% |
| orientation_penalty | -0.188116 | -0.1% | 0.1% | 1.9% |
| speed_penalty | -0.149573 | -0.1% | 0.1% | 3.3% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
