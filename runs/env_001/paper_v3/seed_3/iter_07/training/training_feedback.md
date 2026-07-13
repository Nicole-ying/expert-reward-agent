# Training Feedback

## Final-policy outcome
score=34.640081, len=955.150000, terminated=5/20, truncated=15/20, reward_errors=0
score_range=[-34.721914, 194.726859]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_reward | 60.300000 | 79.3% | 79.3% | 2.1% |
| approach_improvement | 13.823480 | 18.2% | 18.2% | 79.2% |
| soft_landing_velocity | -1.373955 | -1.8% | 1.8% | 99.9% |
| upright_stabilization | -0.523169 | -0.7% | 0.7% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
