# Training Feedback

## Final-policy outcome
score=144.098638, len=935.200000, terminated=2/20, truncated=18/20, reward_errors=0
score_range=[104.995303, 268.013024]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_incentive | 512.018806 | 98.6% | 98.6% | 69.1% |
| progress_reward | 6.774126 | 1.3% | 1.4% | 100.0% |
| angvel_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
