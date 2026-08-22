# Training Feedback

## Final-policy outcome
score=64.238903, len=924.050000, terminated=10/20, truncated=10/20, reward_errors=0
score_range=[-138.260553, 191.986938]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact | 305.309309 | 97.8% | 97.8% | 12.0% |
| pos | 3.554315 | 1.1% | 1.4% | 99.9% |
| vel | -1.495697 | -0.5% | 0.5% | 99.8% |
| angle | -0.546160 | -0.2% | 0.2% | 100.0% |
| angvel | -0.372968 | -0.1% | 0.1% | 99.2% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
