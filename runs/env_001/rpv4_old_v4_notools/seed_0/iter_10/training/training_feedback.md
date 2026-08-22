# Training Feedback

## Final-policy outcome
score=-68.135845, len=70.800000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-110.561152, -24.529183]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| proximity_penalty | -68.338772 | -93.6% | 93.6% | 100.0% |
| vel_penalty | -4.449025 | -6.1% | 6.1% | 17.9% |
| contact_bonus | 0.110533 | 0.2% | 0.2% | 0.8% |
| attitude_penalty | -0.078970 | -0.1% | 0.1% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 17/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
