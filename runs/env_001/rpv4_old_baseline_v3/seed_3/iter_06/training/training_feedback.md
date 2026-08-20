# Training Feedback

## Final-policy outcome
score=-246.827689, len=225.600000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-354.009085, -192.289762]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| velocity_penalty | -17.276114 | -36.0% | 36.0% | 32.7% |
| progress_reward | -5.031623 | -10.5% | 35.2% | 100.0% |
| orientation_penalty | -7.335308 | -15.3% | 15.3% | 100.0% |
| fuel_penalty | -6.502500 | -13.5% | 13.5% | 57.6% |
| stable_contact_reward | 0.000000 | 0.0% | 0.0% | 0.0% |
| touchdown_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 5/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
