# Training Feedback

## Final-policy outcome
score=253.033000, len=786.750000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[34.584905, 315.496631]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| gate_factor | 786.012151 | 45.7% | 45.7% | 100.0% |
| forward_reward | 455.871574 | 26.5% | 26.5% | 99.2% |
| gated_forward | 455.423736 | 26.5% | 26.5% | 99.2% |
| energy_penalty | -23.331972 | -1.4% | 1.4% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
