# Training Feedback

## Final-policy outcome
score=-115.609624, len=68.400000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-139.097838, -99.292225]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress | 1.119159 | 63.6% | 65.9% | 100.0% |
| soft_landing_bonus | 0.500000 | 28.4% | 28.4% | 0.4% |
| stability_penalty | -0.099744 | -5.7% | 5.7% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
