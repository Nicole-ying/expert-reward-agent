# Training Feedback

## Final-policy outcome
score=-52.455098, len=243.150000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-86.853114, -20.064193]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 71.658283 | 95.6% | 95.6% | 97.9% |
| energy_penalty | -2.915391 | -3.9% | 3.9% | 100.0% |
| hinge_penalty | -0.356817 | -0.5% | 0.5% | 1.3% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 3/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
