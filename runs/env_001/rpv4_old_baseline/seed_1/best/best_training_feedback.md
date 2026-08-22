# Training Feedback

## Final-policy outcome
score=-5.048650, len=888.300000, terminated=5/20, truncated=15/20, reward_errors=0
score_range=[-153.494786, 209.210200]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| approach_bonus | 345.025000 | 43.9% | 43.9% | 77.7% |
| shaping | 14.055604 | 1.8% | 27.7% | 100.0% |
| fuel_penalty | -81.885500 | -10.4% | 10.4% | 94.2% |
| step_penalty | -44.415000 | -5.7% | 5.7% | 100.0% |
| crash_penalty | -43.000000 | -5.5% | 5.5% | 0.5% |
| descending_penalty | -32.465837 | -4.1% | 4.1% | 65.4% |
| angle_penalty | -19.011513 | -2.4% | 2.4% | 100.0% |
| angvel_penalty | -1.448646 | -0.2% | 0.2% | 100.0% |
| contact_continuous | 0.691984 | 0.1% | 0.1% | 1.0% |
| success_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
