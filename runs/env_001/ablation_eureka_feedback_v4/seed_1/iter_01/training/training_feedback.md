# Training Feedback

## Final-policy outcome
score=39.495645, len=713.900000, terminated=9/20, truncated=11/20, reward_errors=0
score_range=[-41.941619, 221.519544]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_bonus | 147.500000 | 88.8% | 88.8% | 2.1% |
| proximity_reward | -15.988581 | -9.6% | 9.6% | 100.0% |
| orientation_penalty | -1.683627 | -1.0% | 1.0% | 100.0% |
| velocity_penalty | -0.995888 | -0.6% | 0.6% | 99.9% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
