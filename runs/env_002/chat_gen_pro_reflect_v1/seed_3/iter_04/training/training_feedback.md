# Training Feedback

## Final-policy outcome
score=-74.290111, len=1600.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-91.429523, -64.633436]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| gait_quality | 1200.150000 | 75.1% | 75.1% | 75.0% |
| gait_bonus | 360.045000 | 22.5% | 22.5% | 75.0% |
| both_off_ground_penalty | -36.700000 | -2.3% | 2.3% | 11.5% |
| balance_penalty | -0.132109 | -0.0% | 0.0% | 18.7% |
| forward_progress | 0.124260 | 0.0% | 0.0% | 2.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
