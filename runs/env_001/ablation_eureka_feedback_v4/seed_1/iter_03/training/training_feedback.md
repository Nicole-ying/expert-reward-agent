# Training Feedback

## Final-policy outcome
score=-9.339759, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-44.513305, 20.918004]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| proximity_reward | -13.996765 | -86.4% | 86.4% | 100.0% |
| orientation_penalty | -1.228548 | -7.6% | 7.6% | 100.0% |
| velocity_penalty | -0.968911 | -6.0% | 6.0% | 100.0% |
| contact_reward | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
