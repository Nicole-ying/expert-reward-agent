# Training Feedback

## Final-policy outcome
score=-7.609767, len=869.050000, terminated=4/20, truncated=16/20, reward_errors=0
score_range=[-44.492714, 125.931091]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| proximity_reward | -51.784603 | -77.2% | 77.2% | 100.0% |
| safe_contact_bonus | 9.944489 | 14.8% | 14.8% | 0.7% |
| velocity_penalty | -4.918245 | -7.3% | 7.3% | 99.9% |
| orientation_penalty | -0.420377 | -0.6% | 0.6% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
