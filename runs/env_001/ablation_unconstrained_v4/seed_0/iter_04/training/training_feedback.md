# Training Feedback

## Final-policy outcome
score=14.056370, len=264.100000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-142.530134, 285.530217]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_bonus | 298.000000 | 52.7% | 52.7% | 2.8% |
| settle_reward | 201.652282 | 35.7% | 35.7% | 100.0% |
| distance_penalty | -37.122615 | -6.6% | 6.6% | 100.0% |
| progress_delta | 7.095325 | 1.3% | 2.1% | 99.2% |
| velocity_damping | -11.224114 | -2.0% | 2.0% | 99.9% |
| engine_penalty | -4.885500 | -0.9% | 0.9% | 61.7% |
| orientation_penalty | -0.286798 | -0.1% | 0.1% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
