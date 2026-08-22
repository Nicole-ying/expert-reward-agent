# Training Feedback

## Final-policy outcome
score=286.164528, len=1223.550000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[284.650649, 288.871942]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_progress | 745.438090 | 92.5% | 92.5% | 100.0% |
| upright_stability | -60.534178 | -7.5% | 7.5% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
