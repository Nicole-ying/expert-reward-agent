# Training Feedback

## Final-policy outcome
score=210.908647, len=293.250000, terminated=19/20, truncated=1/20, reward_errors=0
score_range=[-29.902595, 310.610167]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_approach | 34.295316 | 70.9% | 70.9% | 97.9% |
| time_penalty | -5.865000 | -12.1% | 12.1% | 100.0% |
| engine_penalty | -3.813500 | -7.9% | 7.9% | 55.6% |
| distance_progress | 2.374776 | 4.9% | 6.0% | 96.1% |
| stability_penalty | -1.503495 | -3.1% | 3.1% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
