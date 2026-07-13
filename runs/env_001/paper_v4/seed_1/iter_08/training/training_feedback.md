# Training Feedback

## Final-policy outcome
score=-31.051550, len=76.450000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-72.228187, 12.175979]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| shaped_progress | 5.258439 | 68.4% | 71.7% | 99.9% |
| landing_bonus | 1.499009 | 19.5% | 19.5% | 1.0% |
| angular_vel_penalty | -0.674864 | -8.8% | 8.8% | 99.5% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 3/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
