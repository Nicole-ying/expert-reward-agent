# Training Feedback

## Final-policy outcome
score=-51.020801, len=72.400000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-101.254818, -24.178921]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| soft_landing_bonus | 70.000000 | 71.6% | 71.6% | 0.5% |
| proximity_progress | 22.172338 | 22.7% | 23.7% | 100.0% |
| touchdown_reward | 1.779902 | 1.8% | 1.8% | 0.6% |
| speed_penalty | -1.722602 | -1.8% | 1.8% | 99.9% |
| attitude_penalty | -1.101576 | -1.1% | 1.1% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 9/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
