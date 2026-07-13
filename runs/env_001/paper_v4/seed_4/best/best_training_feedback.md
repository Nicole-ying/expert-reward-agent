# Training Feedback

## Final-policy outcome
score=206.135560, len=681.450000, terminated=10/20, truncated=10/20, reward_errors=0
score_range=[127.443360, 288.929483]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| safe_contact_reward | 294.208533 | 94.6% | 94.6% | 51.4% |
| fuel_penalty | -9.771000 | -3.1% | 3.1% | 71.7% |
| stability_penalty | -4.337368 | -1.4% | 1.4% | 100.0% |
| progress_reward | 2.768706 | 0.9% | 0.9% | 99.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
