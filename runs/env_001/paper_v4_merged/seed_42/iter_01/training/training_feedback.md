# Training Feedback

## Final-policy outcome
score=-92.752483, len=68.750000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-113.576725, -63.922293]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| stable_bonus | 1.455661 | 53.9% | 53.9% | 16.9% |
| goal_progress | 1.117861 | 41.4% | 42.9% | 100.0% |
| fuel_penalty | -0.086000 | -3.2% | 3.2% | 12.5% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
