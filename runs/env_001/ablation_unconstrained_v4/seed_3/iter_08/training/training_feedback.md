# Training Feedback

## Final-policy outcome
score=-5510.929150, len=586.850000, terminated=17/20, truncated=3/20, reward_errors=0
score_range=[-10326.028906, -707.312998]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| upright_penalty | -2.760102 | -60.0% | 60.0% | 100.0% |
| approach_reward | 1.839699 | 40.0% | 40.0% | 5.9% |
| soft_landing_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 2/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
