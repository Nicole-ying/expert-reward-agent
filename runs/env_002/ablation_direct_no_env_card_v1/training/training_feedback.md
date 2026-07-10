# Training Feedback

## Final-policy outcome
score=286.119370, len=851.200000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[80.836639, 309.074798]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 485.136249 | 97.0% | 97.1% | 100.0% |
| action_penalty | -14.527097 | -2.9% | 2.9% | 100.0% |
| posture_penalty | -0.210676 | -0.0% | 0.0% | 0.4% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
