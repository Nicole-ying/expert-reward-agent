# Training Feedback

## Final-policy outcome
score=-434.871335, len=967.200000, terminated=1/20, truncated=19/20, reward_errors=0
score_range=[-901.456076, -164.536235]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_proximity_reward | 25.541566 | 60.5% | 60.5% | 68.3% |
| progress_reward | -1.975256 | -4.7% | 37.4% | 100.0% |
| orientation_penalty | -0.874189 | -2.1% | 2.1% | 100.0% |
| vel_damping | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
