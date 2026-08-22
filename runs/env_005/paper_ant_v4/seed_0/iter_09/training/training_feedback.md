# Training Feedback

## Final-policy outcome
score=52.688241, len=52.250000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[0.816130, 182.271411]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 1180.614735 | 77.3% | 77.3% | 75.9% |
| ang_penalty | -204.837459 | -13.4% | 13.4% | 100.0% |
| lat_penalty | -111.982058 | -7.3% | 7.3% | 100.0% |
| height_penalty | -27.625244 | -1.8% | 1.8% | 8.0% |
| upright_penalty | -1.406023 | -0.1% | 0.1% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
