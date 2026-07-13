# Training Feedback

## Final-policy outcome
score=10.283421, len=915.300000, terminated=2/20, truncated=18/20, reward_errors=0
score_range=[-117.732858, 62.701862]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_event | 1719.107382 | 98.5% | 98.5% | 8.2% |
| landing_stability | 15.622014 | 0.9% | 0.9% | 36.7% |
| approach_velocity_penalty | -7.585091 | -0.4% | 0.4% | 100.0% |
| orientation_penalty | -1.678197 | -0.1% | 0.1% | 100.0% |
| progress | 1.196658 | 0.1% | 0.1% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 1/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
