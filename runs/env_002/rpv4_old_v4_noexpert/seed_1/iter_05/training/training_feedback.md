# Training Feedback

## Final-policy outcome
score=291.958633, len=1217.600000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[284.408431, 293.445655]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_velocity_reward | 809.393167 | 99.8% | 99.8% | 99.2% |
| upright_stability_penalty | -1.429529 | -0.2% | 0.2% | 26.5% |
| vertical_oscillation_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
