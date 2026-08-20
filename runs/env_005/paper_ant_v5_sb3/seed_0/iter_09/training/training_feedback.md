# Training Feedback

## Final-policy outcome
score=121.989223, len=357.500000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-4.678686, 368.042796]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward | 563.535246 | 97.6% | 99.9% | 100.0% |
| upright_pen | -0.655226 | -0.1% | 0.1% | 2.9% |
| height_pen | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
