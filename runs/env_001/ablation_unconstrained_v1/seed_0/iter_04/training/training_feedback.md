# Training Feedback

## Final-policy outcome
score=-19.280287, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-43.590654, 18.747831]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| proximity_bonus | 147.608971 | 88.2% | 88.2% | 100.0% |
| stability_penalty | -9.726262 | -5.8% | 5.8% | 100.0% |
| progress_reward | 5.891208 | 3.5% | 4.1% | 100.0% |
| time_penalty | -3.000000 | -1.8% | 1.8% | 100.0% |
| landing_proxy | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
