# Training Feedback

## Final-policy outcome
score=-11.598658, len=80.750000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-71.296190, 49.223280]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_success_bonus | 30.000000 | 49.6% | 49.6% | 0.2% |
| progress | 22.441013 | 37.1% | 38.3% | 100.0% |
| survival | -6.460000 | -10.7% | 10.7% | 100.0% |
| stability | -0.471286 | -0.8% | 0.8% | 100.0% |
| fuel | -0.392000 | -0.6% | 0.6% | 48.5% |
| failure_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 3/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
