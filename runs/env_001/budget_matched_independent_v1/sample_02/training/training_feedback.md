# Training Feedback

## Final-policy outcome
score=228.998179, len=533.650000, terminated=14/20, truncated=6/20, reward_errors=0
score_range=[127.305833, 302.714498]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| soft_landing_bonus | 115.213899 | 93.0% | 93.0% | 43.5% |
| velocity_damping | -6.832732 | -5.5% | 5.5% | 99.9% |
| distance_progress | 1.387302 | 1.1% | 1.2% | 98.1% |
| orientation_penalty | -0.410951 | -0.3% | 0.3% | 99.9% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
