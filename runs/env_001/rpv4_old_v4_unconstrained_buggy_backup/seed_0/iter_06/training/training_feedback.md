# Training Feedback

## Final-policy outcome
score=-124.571050, len=68.350000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-148.794243, -104.867050]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress_reward | 13.372057 | 65.6% | 68.0% | 100.0% |
| soft_approach_reward | 4.742454 | 23.3% | 23.3% | 100.0% |
| orientation_penalty | -1.286041 | -6.3% | 6.3% | 100.0% |
| landing_bonus | 0.390868 | 1.9% | 1.9% | 1.5% |
| action_penalty | -0.110000 | -0.5% | 0.5% | 3.2% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
