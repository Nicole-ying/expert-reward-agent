# Training Feedback

## Final-policy outcome
score=-1471.587872, len=435.400000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-2080.965220, -258.626108]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| soft_success | 12716.349643 | 95.7% | 95.7% | 96.2% |
| velocity_penalty | -376.896537 | -2.8% | 2.8% | 100.0% |
| goal_proximity | 163.929425 | 1.2% | 1.2% | 100.0% |
| thrust_penalty | -21.770000 | -0.2% | 0.2% | 100.0% |
| angle_penalty | -3.785140 | -0.0% | 0.0% | 100.0% |
| contact_quality | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 2/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
