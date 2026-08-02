# Training Feedback

## Final-policy outcome
score=-95.666802, len=71.600000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-151.573446, 18.789616]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| gate_factor | 62.724869 | 65.8% | 65.8% | 100.0% |
| success_bonus | 30.000000 | 31.5% | 31.5% | 0.2% |
| shaping | 2.071662 | 2.2% | 2.3% | 100.0% |
| action_cost | -0.468000 | -0.5% | 0.5% | 32.7% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 14/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
