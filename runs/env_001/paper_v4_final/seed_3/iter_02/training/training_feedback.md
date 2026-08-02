# Training Feedback

## Final-policy outcome
score=-103.621729, len=68.700000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-125.881582, -80.458117]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress_gated | 10.572382 | 65.2% | 67.6% | 100.0% |
| lateral_drift_penalty | -2.917951 | -18.0% | 18.0% | 99.7% |
| landing_bonus | 1.257927 | 7.8% | 7.8% | 0.7% |
| angvel_penalty | -1.077620 | -6.6% | 6.6% | 99.3% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
