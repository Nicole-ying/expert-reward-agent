# Training Feedback

## Final-policy outcome
score=-107.514474, len=798.150000, terminated=12/20, truncated=8/20, reward_errors=0
score_range=[-194.975052, 76.348102]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_bonus | 925.000000 | 49.7% | 49.7% | 1.2% |
| contact_reward | 659.000000 | 35.4% | 35.4% | 4.3% |
| engine_penalty | -135.396000 | -7.3% | 7.3% | 70.9% |
| shaping_reward | 59.510313 | 3.2% | 7.1% | 100.0% |
| time_penalty | -7.981500 | -0.4% | 0.4% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
