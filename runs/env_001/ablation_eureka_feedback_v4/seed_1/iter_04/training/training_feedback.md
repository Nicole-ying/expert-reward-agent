# Training Feedback

## Final-policy outcome
score=109.844683, len=842.750000, terminated=4/20, truncated=16/20, reward_errors=0
score_range=[-9.845876, 302.235811]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_bonus | 66.400000 | 83.8% | 83.8% | 7.9% |
| proximity_reward | -9.382857 | -11.8% | 11.8% | 100.0% |
| orientation_penalty | -2.227679 | -2.8% | 2.8% | 100.0% |
| velocity_penalty | -1.181271 | -1.5% | 1.5% | 99.8% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
