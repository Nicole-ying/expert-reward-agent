# Training Feedback

## Final-policy outcome
score=-131.936383, len=465.400000, terminated=19/20, truncated=1/20, reward_errors=0
score_range=[-206.325637, -8.510957]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| height_cost | -25.575345 | -35.9% | 35.9% | 100.0% |
| contact_reward | 19.146829 | 26.9% | 26.9% | 0.2% |
| speed_cost | -12.321805 | -17.3% | 17.3% | 100.0% |
| progress | 4.375258 | 6.1% | 16.1% | 100.0% |
| vy_cost | -2.212283 | -3.1% | 3.1% | 34.0% |
| engine_penalty | -0.454650 | -0.6% | 0.6% | 97.7% |
| orientation_cost | -0.020376 | -0.0% | 0.0% | 100.0% |
| landing_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
