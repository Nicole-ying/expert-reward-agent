# Training Feedback

## Final-policy outcome
score=-124.549203, len=68.400000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-151.335003, -85.186092]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| failure_penalty | -17.000000 | -52.8% | 52.8% | 2.5% |
| success_reward | 10.000000 | 31.1% | 31.1% | 0.1% |
| progress | 3.335475 | 10.4% | 10.7% | 100.0% |
| stability_penalty | -1.588632 | -4.9% | 4.9% | 100.0% |
| action_penalty | -0.072500 | -0.2% | 0.2% | 2.1% |
| soft_landing | 0.051314 | 0.2% | 0.2% | 0.3% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
