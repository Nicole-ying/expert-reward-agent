# Training Feedback

## Final-policy outcome
score=147.707809, len=369.600000, terminated=13/20, truncated=7/20, reward_errors=0
score_range=[4.678243, 660.127832]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 1523.117973 | 54.5% | 54.5% | 40.3% |
| upright_penalty | -812.660031 | -29.1% | 29.1% | 100.0% |
| ang_penalty | -305.616244 | -10.9% | 10.9% | 88.3% |
| lat_penalty | -118.341091 | -4.2% | 4.2% | 88.3% |
| height_penalty | -34.676276 | -1.2% | 1.2% | 12.1% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
