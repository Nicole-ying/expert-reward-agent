# Training Feedback

## Final-policy outcome
score=259.503249, len=290.300000, terminated=19/20, truncated=1/20, reward_errors=0
score_range=[126.026450, 298.536302]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| grounded_quality | 28.075227 | 37.9% | 37.9% | 35.3% |
| velocity_penalty | -18.884340 | -25.5% | 25.5% | 60.4% |
| landing_quality | 17.892238 | 24.1% | 24.1% | 100.0% |
| approach_progress | 6.554560 | 8.8% | 9.1% | 95.7% |
| angle_stability | -2.546881 | -3.4% | 3.4% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
