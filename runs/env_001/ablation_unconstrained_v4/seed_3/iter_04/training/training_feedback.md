# Training Feedback

## Final-policy outcome
score=-123.191393, len=80.100000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-197.504246, 56.357220]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_bonus | 6.000000 | 40.6% | 40.6% | 0.7% |
| descent_reward | -5.223363 | -35.3% | 35.3% | 82.3% |
| contact_reward | 2.598752 | 17.6% | 17.6% | 6.8% |
| engine_penalty | -0.960000 | -6.5% | 6.5% | 6.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 17/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
