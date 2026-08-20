# Training Feedback

## Final-policy outcome
score=99.047748, len=817.150000, terminated=14/20, truncated=6/20, reward_errors=0
score_range=[-154.832997, 259.513193]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| engine_cost | -11.599000 | -41.1% | 41.1% | 71.0% |
| progress | 6.077497 | 21.5% | 25.9% | 99.0% |
| attitude_penalty | -4.627532 | -16.4% | 16.4% | 100.0% |
| landing_quality | 2.589379 | 9.2% | 9.2% | 0.2% |
| landing_velocity_penalty | -2.090656 | -7.4% | 7.4% | 4.1% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
