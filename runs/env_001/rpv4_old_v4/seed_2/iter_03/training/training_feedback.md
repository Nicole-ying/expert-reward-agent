# Training Feedback

## Final-policy outcome
score=-73.586869, len=96.300000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-152.380896, 0.221844]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress | 6.153739 | 52.8% | 55.9% | 100.0% |
| contact_guidance | 3.786319 | 32.5% | 32.5% | 0.9% |
| orientation_penalty | -0.902325 | -7.7% | 7.7% | 100.0% |
| velocity_penalty | -0.444880 | -3.8% | 3.8% | 99.8% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 12/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
