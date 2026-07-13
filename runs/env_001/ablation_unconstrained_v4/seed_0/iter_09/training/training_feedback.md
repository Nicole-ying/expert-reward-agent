# Training Feedback

## Final-policy outcome
score=-115.081713, len=68.450000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-140.076175, -93.973586]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress_delta | 44.785259 | 50.1% | 51.8% | 100.0% |
| distance_penalty | -33.270019 | -37.2% | 37.2% | 100.0% |
| velocity_damping | -5.171897 | -5.8% | 5.8% | 100.0% |
| stable_landing_bonus | 4.178748 | 4.7% | 4.7% | 0.6% |
| engine_penalty | -0.287500 | -0.3% | 0.3% | 8.4% |
| orientation_penalty | -0.234627 | -0.3% | 0.3% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
