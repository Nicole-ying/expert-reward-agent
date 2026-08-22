# Training Feedback

## Final-policy outcome
score=-117.594935, len=68.400000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-142.928717, -92.099139]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| survival_penalty | -34.200000 | -39.6% | 39.6% | 100.0% |
| soft_contact_reward | 29.530681 | 34.2% | 34.2% | 1.5% |
| progress_reward | -12.358456 | -14.3% | 14.3% | 100.0% |
| orientation_penalty | -10.011325 | -11.6% | 11.6% | 100.0% |
| engine_penalty | -0.268500 | -0.3% | 0.3% | 13.1% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
