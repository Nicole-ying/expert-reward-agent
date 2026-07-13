# Training Feedback

## Final-policy outcome
score=-51.015629, len=974.750000, terminated=1/20, truncated=19/20, reward_errors=0
score_range=[-93.387214, -7.111848]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| goal_proximity | -258.114002 | -92.2% | 92.2% | 100.0% |
| velocity_penalty | -14.016724 | -5.0% | 5.0% | 100.0% |
| contact_reward | 6.831231 | 2.4% | 2.4% | 0.5% |
| orientation_penalty | -0.884365 | -0.3% | 0.3% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
