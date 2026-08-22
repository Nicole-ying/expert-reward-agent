# Training Feedback

## Final-policy outcome
score=-203.446380, len=154.150000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-270.266370, -158.776888]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress_shaping | 1.487613 | 17.9% | 52.6% | 100.0% |
| angle_penalty | -2.459905 | -29.7% | 29.7% | 100.0% |
| action_cost | -1.472000 | -17.8% | 17.8% | 47.7% |
| crash_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |
| success_landing_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 11/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
