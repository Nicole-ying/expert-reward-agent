# Training Feedback

## Final-policy outcome
score=-57.363366, len=628.500000, terminated=15/20, truncated=5/20, reward_errors=0
score_range=[-201.258549, 140.281831]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_proxy | 1236.415922 | 78.0% | 78.0% | 100.0% |
| distance_penalty | -335.273623 | -21.1% | 21.1% | 100.0% |
| orientation_penalty | -9.655049 | -0.6% | 0.6% | 100.0% |
| velocity_penalty | -4.294234 | -0.3% | 0.3% | 100.0% |
| success_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
