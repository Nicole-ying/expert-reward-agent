# Training Feedback

## Final-policy outcome
score=89.156982, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[59.816231, 135.415547]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_quality | 7.113571 | 44.2% | 44.2% | 100.0% |
| velocity_penalty | -4.623635 | -28.7% | 28.7% | 4.9% |
| approach_progress | 2.679012 | 16.6% | 18.3% | 100.0% |
| angle_stability | -1.403073 | -8.7% | 8.7% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
