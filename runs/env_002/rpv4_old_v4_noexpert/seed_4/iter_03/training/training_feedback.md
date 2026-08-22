# Training Feedback

## Final-policy outcome
score=240.531626, len=1385.900000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-7.624346, 282.012835]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_progress | 861.500221 | 90.2% | 90.2% | 100.0% |
| upright_stability | -72.991843 | -7.6% | 7.6% | 100.0% |
| energy_efficiency_penalty | -20.704270 | -2.2% | 2.2% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
