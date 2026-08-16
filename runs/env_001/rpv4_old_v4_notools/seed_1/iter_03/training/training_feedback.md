# Training Feedback

## Final-policy outcome
score=-10.421864, len=487.750000, terminated=15/20, truncated=5/20, reward_errors=0
score_range=[-214.660892, 241.448691]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| soft_landing_bonus | 320.483435 | 88.8% | 88.8% | 18.5% |
| proximity_progress | 22.123964 | 6.1% | 10.8% | 99.7% |
| attitude_penalty | -1.342851 | -0.4% | 0.4% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 1/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
