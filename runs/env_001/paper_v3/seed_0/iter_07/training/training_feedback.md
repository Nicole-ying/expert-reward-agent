# Training Feedback

## Final-policy outcome
score=96.514091, len=382.150000, terminated=16/20, truncated=4/20, reward_errors=0
score_range=[-166.151522, 239.486188]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_proxy | 430.114292 | 65.5% | 65.5% | 46.0% |
| distance_penalty | -206.423812 | -31.4% | 31.4% | 100.0% |
| velocity_penalty | -10.854147 | -1.7% | 1.7% | 99.6% |
| orientation_penalty | -9.360479 | -1.4% | 1.4% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
