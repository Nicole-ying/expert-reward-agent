# Training Feedback

## Final-policy outcome
score=-24.389467, len=919.600000, terminated=4/20, truncated=16/20, reward_errors=0
score_range=[-106.995986, 189.654465]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| touchdown_reward | 51.135346 | 59.8% | 59.8% | 2.1% |
| proximity_progress | 20.576698 | 24.1% | 36.0% | 100.0% |
| attitude_penalty | -2.690323 | -3.1% | 3.1% | 100.0% |
| speed_penalty | -0.896170 | -1.0% | 1.0% | 100.0% |
| landing_quality_reward | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
