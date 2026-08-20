# Training Feedback

## Final-policy outcome
score=-2.302470, len=950.100000, terminated=2/20, truncated=18/20, reward_errors=0
score_range=[-39.488471, 191.085708]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| approach_improvement | 2.348736 | 25.9% | 35.5% | 100.0% |
| soft_landing_velocity | -3.080972 | -34.0% | 34.0% | 100.0% |
| contact_reward | 2.250000 | 24.8% | 24.8% | 0.1% |
| upright_stabilization | -0.517143 | -5.7% | 5.7% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
