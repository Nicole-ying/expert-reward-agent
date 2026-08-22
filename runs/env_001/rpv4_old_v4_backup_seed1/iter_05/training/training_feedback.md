# Training Feedback

## Final-policy outcome
score=-33.040122, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-67.620306, 12.216166]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| velocity_penalty | -10.121546 | -50.6% | 50.6% | 100.0% |
| goal_proximity | 4.285993 | 21.4% | 31.4% | 100.0% |
| upright_penalty | -3.598494 | -18.0% | 18.0% | 100.0% |
| contact_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
