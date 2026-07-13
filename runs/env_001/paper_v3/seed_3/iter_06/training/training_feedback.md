# Training Feedback

## Final-policy outcome
score=-31.485462, len=960.650000, terminated=3/20, truncated=17/20, reward_errors=0
score_range=[-114.582321, 95.587091]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| approach_and_soft_landing | 686.965146 | 99.1% | 99.1% | 100.0% |
| contact_reward | 5.850000 | 0.8% | 0.8% | 0.2% |
| upright_stabilization | -0.621847 | -0.1% | 0.1% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
