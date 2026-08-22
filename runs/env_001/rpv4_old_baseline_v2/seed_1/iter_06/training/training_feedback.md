# Training Feedback

## Final-policy outcome
score=-47.425882, len=75.050000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-93.804292, -4.433477]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_reward | 76.000000 | 62.8% | 62.8% | 2.1% |
| distance_penalty | -34.506254 | -28.5% | 28.5% | 100.0% |
| vel_penalty | -8.282890 | -6.8% | 6.8% | 100.0% |
| success_bonus | 1.400000 | 1.2% | 1.2% | 0.9% |
| angvel_penalty | -0.518872 | -0.4% | 0.4% | 99.9% |
| angle_penalty | -0.323311 | -0.3% | 0.3% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 8/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
