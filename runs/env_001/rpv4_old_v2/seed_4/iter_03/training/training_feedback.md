# Training Feedback

## Final-policy outcome
score=203.999462, len=569.650000, terminated=18/20, truncated=2/20, reward_errors=0
score_range=[68.170546, 245.278434]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_completion | 363.115059 | 79.1% | 79.1% | 16.5% |
| soft_landing_speed_penalty | -71.365811 | -15.6% | 15.6% | 99.8% |
| approach_progress | 19.318184 | 4.2% | 4.5% | 97.8% |
| orientation_penalty | -3.520726 | -0.8% | 0.8% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
