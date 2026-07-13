# Training Feedback

## Final-policy outcome
score=-123.403390, len=68.400000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-147.825372, -100.787007]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| proximity | 1.134819 | 63.4% | 63.4% | 92.0% |
| landing_event | 0.624726 | 34.9% | 34.9% | 0.1% |
| energy_penalty | -0.029000 | -1.6% | 1.6% | 4.2% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
