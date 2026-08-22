# Training Feedback

## Final-policy outcome
score=-661.901706, len=473.800000, terminated=13/20, truncated=7/20, reward_errors=0
score_range=[-1914.957840, -113.363910]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| height_factor | 371.419837 | 47.5% | 47.5% | 100.0% |
| upright_factor | 292.975728 | 37.5% | 37.5% | 62.4% |
| forward_speed_reward | 117.182968 | 15.0% | 15.0% | 46.7% |
| height_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |
| upright_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 6/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
