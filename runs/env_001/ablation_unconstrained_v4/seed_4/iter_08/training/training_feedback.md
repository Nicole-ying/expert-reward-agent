# Training Feedback

## Final-policy outcome
score=-59.385433, len=988.900000, terminated=1/20, truncated=19/20, reward_errors=0
score_range=[-130.427246, -7.797466]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress | 0.754262 | 76.6% | 97.3% | 100.0% |
| angle_penalty | -0.026602 | -2.7% | 2.7% | 0.2% |
| contact_reward | 0.000000 | 0.0% | 0.0% | 0.0% |
| speed_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
