# Training Feedback

## Final-policy outcome
score=-15.416144, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-51.588635, 19.512293]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress | 6.258317 | 50.6% | 58.0% | 100.0% |
| velocity_penalty | -3.151573 | -25.5% | 25.5% | 100.0% |
| orientation_penalty | -2.040824 | -16.5% | 16.5% | 100.0% |
| settling_quality | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
