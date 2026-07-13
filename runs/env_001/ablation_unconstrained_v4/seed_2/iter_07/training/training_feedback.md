# Training Feedback

## Final-policy outcome
score=-119.598908, len=685.750000, terminated=16/20, truncated=4/20, reward_errors=0
score_range=[-211.465199, 115.345158]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| height_near | 4738.597271 | 94.9% | 94.9% | 100.0% |
| alignment | 219.240640 | 4.4% | 4.4% | 100.0% |
| contact_reward | 16.900000 | 0.3% | 0.3% | 1.2% |
| lat_penalty | -7.154757 | -0.1% | 0.1% | 25.4% |
| progress | 1.469594 | 0.0% | 0.1% | 100.0% |
| att_penalty | -2.408608 | -0.0% | 0.0% | 11.7% |
| angvel_penalty | -1.970244 | -0.0% | 0.0% | 7.7% |
| down_penalty | -0.007635 | -0.0% | 0.0% | 0.4% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
