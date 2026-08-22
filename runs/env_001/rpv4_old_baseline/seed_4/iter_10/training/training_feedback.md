# Training Feedback

## Final-policy outcome
score=-72.162335, len=446.350000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-152.073162, 155.186276]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_reward | 2994.306262 | 96.9% | 96.9% | 4.9% |
| time_penalty | -22.317500 | -0.7% | 0.7% | 100.0% |
| centering_penalty | -17.361525 | -0.6% | 0.6% | 100.0% |
| angle_penalty | -16.398985 | -0.5% | 0.5% | 100.0% |
| engine_penalty | -15.000000 | -0.5% | 0.5% | 89.6% |
| speed_penalty | -13.047093 | -0.4% | 0.4% | 100.0% |
| approach_reward | 2.714465 | 0.1% | 0.3% | 100.0% |
| angvel_penalty | -1.117025 | -0.0% | 0.0% | 99.5% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
