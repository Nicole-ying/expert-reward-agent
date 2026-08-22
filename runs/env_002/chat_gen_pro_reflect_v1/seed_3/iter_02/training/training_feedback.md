# Training Feedback

## Final-policy outcome
score=-49.881089, len=194.450000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-79.360974, -13.406592]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| gait_quality | 113.150000 | 35.1% | 35.1% | 58.2% |
| forward_progress_raw | 102.945381 | 32.0% | 32.0% | 96.5% |
| forward_with_gait | 85.639802 | 26.6% | 26.6% | 96.5% |
| both_off_ground_penalty | -15.450000 | -4.8% | 4.8% | 39.7% |
| balance_penalty | -4.946417 | -1.5% | 1.5% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 2/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
