# Training Feedback

## Final-policy outcome
score=370.674682, len=792.100000, terminated=9/20, truncated=11/20, reward_errors=0
score_range=[-264.678559, 606.290917]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 1930.053090 | 78.3% | 79.7% | 100.0% |
| height_reward | 265.243034 | 10.8% | 10.8% | 100.0% |
| action_cost | -142.729251 | -5.8% | 5.8% | 100.0% |
| lateral_velocity_penalty | -92.072981 | -3.7% | 3.7% | 100.0% |
| height_boundary_penalty | -0.090425 | -0.0% | 0.0% | 0.1% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
