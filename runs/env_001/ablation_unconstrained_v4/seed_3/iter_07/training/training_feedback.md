# Training Feedback

## Final-policy outcome
score=-122.031697, len=68.300000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-143.199345, -99.642128]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_reward | 6.000000 | 56.8% | 56.8% | 0.4% |
| progress_reward | 2.269205 | 21.5% | 21.5% | 92.2% |
| lateral_penalty | -2.108171 | -19.9% | 19.9% | 100.0% |
| engine_penalty | -0.195000 | -1.8% | 1.8% | 5.7% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
