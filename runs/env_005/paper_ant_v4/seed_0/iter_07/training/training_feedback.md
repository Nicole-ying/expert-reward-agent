# Training Feedback

## Final-policy outcome
score=203.897414, len=344.000000, terminated=14/20, truncated=6/20, reward_errors=0
score_range=[6.494502, 722.989021]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 1582.189002 | 44.4% | 44.4% | 39.3% |
| upright_penalty | -1344.900949 | -37.8% | 37.8% | 27.4% |
| ang_penalty | -401.086912 | -11.3% | 11.3% | 68.1% |
| lat_penalty | -163.208277 | -4.6% | 4.6% | 68.0% |
| height_penalty | -69.697536 | -2.0% | 2.0% | 26.4% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
