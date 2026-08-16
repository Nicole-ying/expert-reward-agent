# Training Feedback

## Final-policy outcome
score=-115.610232, len=948.450000, terminated=8/20, truncated=12/20, reward_errors=0
score_range=[-227.490867, 13.995974]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| approach_reward | 12.052421 | 16.0% | 41.1% | 100.0% |
| fuel_penalty | -28.453500 | -37.7% | 37.7% | 100.0% |
| velocity_penalty | -9.035890 | -12.0% | 12.0% | 100.0% |
| close_proximity | 4.308586 | 5.7% | 5.7% | 29.9% |
| angle_penalty | -2.619085 | -3.5% | 3.5% | 100.0% |
| boundary_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |
| contact_reward | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
