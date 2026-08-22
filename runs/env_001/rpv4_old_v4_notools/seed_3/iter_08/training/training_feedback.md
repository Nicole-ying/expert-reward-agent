# Training Feedback

## Final-policy outcome
score=-110.443065, len=68.450000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-126.734188, -95.059083]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| soft_landing_bonus | 0.277499 | 40.2% | 40.2% | 1.7% |
| stability_penalty | -0.213771 | -31.0% | 31.0% | 100.0% |
| contact_alignment | 0.115000 | 16.7% | 16.7% | 1.7% |
| approach_delta_gated | 0.070212 | 10.2% | 12.2% | 100.0% |
| energy_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
