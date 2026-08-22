# Training Feedback

## Final-policy outcome
score=-158.859613, len=609.350000, terminated=12/20, truncated=8/20, reward_errors=0
score_range=[-1130.145519, 214.657723]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward | 535.146233 | 44.4% | 46.4% | 100.0% |
| height_factor | 543.552077 | 45.1% | 45.1% | 100.0% |
| upright_penalty | -78.955896 | -6.6% | 6.6% | 28.4% |
| lateral_penalty | -12.420444 | -1.0% | 1.0% | 42.8% |
| vertical_penalty | -10.144040 | -0.8% | 0.8% | 33.2% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
