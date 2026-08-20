# Training Feedback

## Final-policy outcome
score=281.820049, len=1036.400000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[16.907162, 304.671329]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| gate_factor | 1927.051562 | 57.7% | 57.7% | 100.0% |
| gated_forward | 900.076313 | 26.9% | 26.9% | 99.8% |
| forward_reward | 484.644203 | 14.5% | 14.5% | 99.8% |
| energy_penalty | -28.607421 | -0.9% | 0.9% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
