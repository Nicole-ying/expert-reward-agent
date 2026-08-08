# Training Feedback

## Final-policy outcome
score=-61.629152, len=240.350000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-107.196828, -25.005159]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| vertical_gate | 229.654922 | 32.6% | 32.6% | 100.0% |
| health_gate | 206.672514 | 29.4% | 29.4% | 100.0% |
| angle_gate | 189.777952 | 27.0% | 27.0% | 99.1% |
| progress_raw | 75.989850 | 10.8% | 11.0% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 3/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
