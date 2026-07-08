# Training Feedback

## Final-policy outcome
score=289.987325, len=1026.400000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[287.591273, 293.393597]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 503.950706 | 99.2% | 99.2% | 100.0% |
| angle_penalty | -3.287244 | -0.6% | 0.6% | 100.0% |
| vert_penalty | -0.831285 | -0.2% | 0.2% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
