# Training Feedback

## Final-policy outcome
score=93.865259, len=878.000000, terminated=11/20, truncated=9/20, reward_errors=0
score_range=[-44.108231, 236.790115]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_proxy | 42.611703 | 89.3% | 89.3% | 2.5% |
| velocity_penalty | -3.190549 | -6.7% | 6.7% | 99.1% |
| progress_reward | 1.363556 | 2.9% | 3.3% | 99.1% |
| attitude_penalty | -0.312130 | -0.7% | 0.7% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
