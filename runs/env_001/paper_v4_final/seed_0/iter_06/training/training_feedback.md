# Training Feedback

## Final-policy outcome
score=243.146690, len=386.150000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[212.294419, 275.698508]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| success_bonus | 56.538389 | 76.7% | 76.7% | 26.3% |
| soft_landing | 8.160170 | 11.1% | 11.1% | 70.9% |
| contact_stability | 7.653822 | 10.4% | 10.4% | 26.4% |
| progress_reward | 1.302760 | 1.8% | 1.9% | 96.4% |
| angle_penalty | -0.032117 | -0.0% | 0.0% | 99.9% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
