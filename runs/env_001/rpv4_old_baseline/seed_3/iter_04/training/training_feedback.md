# Training Feedback

## Final-policy outcome
score=-158.525996, len=533.000000, terminated=19/20, truncated=1/20, reward_errors=0
score_range=[-231.888213, -62.208703]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| speed_penalty_near | -36.471318 | -84.1% | 84.1% | 100.0% |
| speed_penalty_global | -3.778586 | -8.7% | 8.7% | 100.0% |
| progress | 0.464312 | 1.1% | 4.3% | 100.0% |
| vert_speed_penalty | -1.241775 | -2.9% | 2.9% | 100.0% |
| orientation | -0.002666 | -0.0% | 0.0% | 100.0% |
| landing | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
