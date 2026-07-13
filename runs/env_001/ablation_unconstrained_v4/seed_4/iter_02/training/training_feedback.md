# Training Feedback

## Final-policy outcome
score=-110.659067, len=68.450000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-129.407665, -78.933292]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress | 5.605463 | 28.1% | 29.1% | 100.0% |
| hspeed_penalty | -5.576290 | -28.0% | 28.0% | 100.0% |
| x_center_penalty | -5.129217 | -25.8% | 25.8% | 100.0% |
| landing_vel_penalty | -2.760965 | -13.9% | 13.9% | 100.0% |
| upright_penalty | -0.649004 | -3.3% | 3.3% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
