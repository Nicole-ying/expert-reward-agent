# Training Feedback

## Final-policy outcome
score=-55.582721, len=304.000000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-89.693171, -10.886341]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| stability_gate | 292.377206 | 57.5% | 57.5% | 99.3% |
| forward_reward | 93.533620 | 18.4% | 18.4% | 95.7% |
| gated_forward | 90.807872 | 17.9% | 17.9% | 95.3% |
| contact_transition_reward | 28.370904 | 5.6% | 5.6% | 99.9% |
| energy_penalty | -2.635955 | -0.5% | 0.5% | 100.0% |
| roughness_penalty | -0.418574 | -0.1% | 0.1% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 2/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
