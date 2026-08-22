# Training Feedback

## Final-policy outcome
score=-307.974882, len=956.800000, terminated=1/20, truncated=19/20, reward_errors=0
score_range=[-348.075695, -134.394434]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_reward | 5647.930624 | 99.9% | 99.9% | 100.0% |
| progress_reward | -3.342327 | -0.1% | 0.1% | 100.0% |
| crash_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 1/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
