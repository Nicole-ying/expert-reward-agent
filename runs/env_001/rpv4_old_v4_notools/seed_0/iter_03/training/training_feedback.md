# Training Feedback

## Final-policy outcome
score=-124.547041, len=592.800000, terminated=19/20, truncated=1/20, reward_errors=0
score_range=[-217.230008, 124.711063]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_reward | 253.754127 | 91.4% | 91.4% | 0.7% |
| vel_damping | -22.307194 | -8.0% | 8.0% | 83.9% |
| orientation_penalty | -1.129505 | -0.4% | 0.4% | 100.0% |
| progress_reward | 0.201174 | 0.1% | 0.2% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
