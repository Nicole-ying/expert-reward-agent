# Training Feedback

## Final-policy outcome
score=407.457281, len=437.700000, terminated=15/20, truncated=5/20, reward_errors=0
score_range=[-1.109121, 1321.372171]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 1927.958824 | 88.8% | 89.6% | 90.1% |
| height_reward | 129.675525 | 6.0% | 6.0% | 100.0% |
| action_cost | -80.616551 | -3.7% | 3.7% | 100.0% |
| upright_penalty | -14.523626 | -0.7% | 0.7% | 7.4% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
