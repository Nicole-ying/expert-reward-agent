# Training Feedback

## Final-policy outcome
score=-117.193023, len=69.750000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-196.679836, 17.571949]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| vel_damping | -17.090170 | -50.1% | 50.1% | 100.0% |
| progress_reward | 13.367969 | 39.2% | 40.6% | 100.0% |
| orientation_penalty | -2.090452 | -6.1% | 6.1% | 100.0% |
| landing_bonus | 0.958299 | 2.8% | 2.8% | 1.5% |
| action_penalty | -0.105000 | -0.3% | 0.3% | 3.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 18/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
