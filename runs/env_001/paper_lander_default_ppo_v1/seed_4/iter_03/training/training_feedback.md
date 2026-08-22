# Training Feedback

## Final-policy outcome
score=187.039155, len=715.900000, terminated=9/20, truncated=11/20, reward_errors=0
score_range=[-20.937976, 318.999076]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| settlement_bonus | 988.591783 | 99.0% | 99.0% | 71.3% |
| speed_penalty_gated | -6.264979 | -0.6% | 0.6% | 99.4% |
| proximity_reward | 2.754324 | 0.3% | 0.3% | 99.3% |
| orientation_penalty | -0.748503 | -0.1% | 0.1% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
