# Training Feedback

## Final-policy outcome
score=59.583215, len=417.600000, terminated=17/20, truncated=3/20, reward_errors=0
score_range=[-140.494405, 231.250382]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| soft_landing | 770.756891 | 94.2% | 94.2% | 12.5% |
| progress_reward | 26.861687 | 3.3% | 4.7% | 100.0% |
| descent_bonus | 7.156978 | 0.9% | 0.9% | 69.3% |
| attitude_penalty | -2.497972 | -0.3% | 0.3% | 100.0% |
| crash_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
