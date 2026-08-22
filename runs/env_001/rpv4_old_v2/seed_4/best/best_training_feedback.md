# Training Feedback

## Final-policy outcome
score=215.504114, len=517.900000, terminated=18/20, truncated=2/20, reward_errors=0
score_range=[79.062130, 258.138414]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_completion | 356.972250 | 83.6% | 83.6% | 18.4% |
| soft_landing_speed_penalty | -60.296305 | -14.1% | 14.1% | 99.8% |
| approach_progress | 6.596181 | 1.5% | 1.6% | 97.7% |
| orientation_penalty | -2.671592 | -0.6% | 0.6% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
