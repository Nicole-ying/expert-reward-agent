# Training Feedback

## Final-policy outcome
score=-82.069513, len=86.100000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-83.198093, -81.292727]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| gate_factor | 145.480884 | 61.9% | 61.9% | 100.0% |
| gated_forward | 53.696182 | 22.9% | 22.9% | 100.0% |
| forward_reward | 30.293336 | 12.9% | 12.9% | 100.0% |
| energy_penalty | -5.347363 | -2.3% | 2.3% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
