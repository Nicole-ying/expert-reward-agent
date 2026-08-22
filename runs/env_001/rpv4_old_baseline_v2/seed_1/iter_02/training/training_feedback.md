# Training Feedback

## Final-policy outcome
score=-60.941087, len=975.750000, terminated=1/20, truncated=19/20, reward_errors=0
score_range=[-102.791515, -15.406515]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| vel_penalty | -5.955435 | -44.6% | 44.6% | 100.0% |
| progress | 1.471255 | 11.0% | 21.0% | 100.0% |
| angle_penalty | -2.304841 | -17.3% | 17.3% | 100.0% |
| contact_reward | 2.050000 | 15.4% | 15.4% | 0.1% |
| angvel_penalty | -0.245465 | -1.8% | 1.8% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
