# Training Feedback

## Final-policy outcome
score=299.047285, len=917.350000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[82.654062, 318.367088]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 3002.693140 | 67.8% | 67.8% | 99.7% |
| balance_gate | 891.049494 | 20.1% | 20.1% | 100.0% |
| gait_quality | 330.800000 | 7.5% | 7.5% | 36.1% |
| gait_efficiency | 86.260431 | 1.9% | 1.9% | 35.8% |
| balance_modulation | -83.970387 | -1.9% | 1.9% | 100.0% |
| both_off_penalty | -25.510000 | -0.6% | 0.6% | 27.8% |
| action_penalty | -11.216696 | -0.3% | 0.3% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
