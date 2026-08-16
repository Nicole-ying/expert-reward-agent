# Training Feedback

## Final-policy outcome
score=39.147304, len=655.550000, terminated=10/20, truncated=10/20, reward_errors=0
score_range=[-67.311211, 278.404267]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_bonus | 183.600000 | 78.6% | 78.6% | 1.9% |
| contact_reward | 26.662500 | 11.4% | 11.4% | 3.5% |
| action_reward | -15.787000 | -6.8% | 6.8% | 65.7% |
| vel_damping | -4.671777 | -2.0% | 2.0% | 21.2% |
| progress_reward | 1.671110 | 0.7% | 1.1% | 99.4% |
| orientation_penalty | -0.389096 | -0.2% | 0.2% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
