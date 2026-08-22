# Training Feedback

## Final-policy outcome
score=-71.398043, len=70.550000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-117.259421, -29.067472]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress_gated | 10.498345 | 64.3% | 66.8% | 100.0% |
| lateral_drift_penalty | -2.968266 | -18.2% | 18.2% | 99.9% |
| landing_bonus | 1.281019 | 7.8% | 7.8% | 0.7% |
| angvel_penalty | -1.176742 | -7.2% | 7.2% | 99.6% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 13/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
