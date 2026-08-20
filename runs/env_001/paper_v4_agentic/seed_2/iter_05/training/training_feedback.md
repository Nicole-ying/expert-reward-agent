# Training Feedback

## Final-policy outcome
score=-115.165245, len=68.400000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-141.582367, -91.736133]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| proximity_delta | 55.947302 | 82.9% | 85.8% | 100.0% |
| velocity_danger | -8.538899 | -12.7% | 12.7% | 100.0% |
| orientation_penalty | -1.025295 | -1.5% | 1.5% | 100.0% |
| soft_approach_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
