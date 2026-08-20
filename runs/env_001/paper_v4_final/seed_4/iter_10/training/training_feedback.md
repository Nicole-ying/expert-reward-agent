# Training Feedback

## Final-policy outcome
score=-17.308492, len=129.900000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-51.203160, 19.977203]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_success_bonus | 120.000000 | 75.1% | 75.1% | 0.6% |
| progress | 25.961909 | 16.2% | 17.0% | 100.0% |
| survival | -10.392000 | -6.5% | 6.5% | 100.0% |
| stability | -1.526689 | -1.0% | 1.0% | 100.0% |
| fuel | -0.737500 | -0.5% | 0.5% | 56.8% |
| failure_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 1/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
