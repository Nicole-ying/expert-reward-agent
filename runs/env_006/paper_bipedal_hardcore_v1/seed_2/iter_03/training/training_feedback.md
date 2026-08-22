# Training Feedback

## Final-policy outcome
score=-95.842155, len=74.800000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-96.019857, -95.640137]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| terrain_gate | 37.423793 | 61.9% | 61.9% | 100.0% |
| terrain_roughness | 16.018375 | 26.5% | 26.5% | 100.0% |
| forward_reward | 6.412768 | 10.6% | 10.7% | 100.0% |
| air_stability_penalty | -0.500731 | -0.8% | 0.8% | 13.3% |
| balance_penalty | -0.058223 | -0.1% | 0.1% | 28.3% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
