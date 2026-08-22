# Training Feedback

## Final-policy outcome
score=-114.532526, len=68.350000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-129.154126, -97.170495]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| height_penalty | -127.442556 | -43.9% | 43.9% | 100.0% |
| landing_bonus | 60.000000 | 20.7% | 20.7% | 0.4% |
| progress | 56.030154 | 19.3% | 20.0% | 100.0% |
| contact_reward | 27.500000 | 9.5% | 9.5% | 3.0% |
| speed_penalty | -17.201291 | -5.9% | 5.9% | 100.0% |
| engine_penalty | -0.035200 | -0.0% | 0.0% | 51.5% |
| orientation | -0.022375 | -0.0% | 0.0% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
