# Training Feedback

## Final-policy outcome
score=-53.584149, len=216.150000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-86.383689, -27.101370]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress | 70.926116 | 96.4% | 96.4% | 98.4% |
| action_penalty | -2.311209 | -3.1% | 3.1% | 100.0% |
| hinge_balance_penalty | -0.352891 | -0.5% | 0.5% | 1.7% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 4/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
