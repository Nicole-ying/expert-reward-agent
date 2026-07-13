# Training Feedback

## Final-policy outcome
score=256.347342, len=351.350000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[229.129917, 291.378163]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_quality | 20.770152 | 41.2% | 41.2% | 100.0% |
| grounded_quality | 13.972010 | 27.7% | 27.7% | 14.8% |
| velocity_penalty | -10.721436 | -21.3% | 21.3% | 53.8% |
| approach_progress | 2.745139 | 5.4% | 5.6% | 95.7% |
| angle_stability | -2.141589 | -4.2% | 4.2% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
