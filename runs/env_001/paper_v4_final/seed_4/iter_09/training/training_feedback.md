# Training Feedback

## Final-policy outcome
score=-41.151239, len=75.000000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-71.929630, -1.401947]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress | 22.245364 | 49.6% | 51.5% | 100.0% |
| landing_success_bonus | 15.000000 | 33.5% | 33.5% | 0.1% |
| survival | -6.000000 | -13.4% | 13.4% | 100.0% |
| stability | -0.428865 | -1.0% | 1.0% | 100.0% |
| fuel | -0.317500 | -0.7% | 0.7% | 42.3% |
| failure_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 6/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
