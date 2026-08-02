# Training Feedback

## Final-policy outcome
score=194.868331, len=661.900000, terminated=16/20, truncated=4/20, reward_errors=0
score_range=[68.690992, 250.993146]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| stable_bonus | 281.108735 | 81.7% | 81.7% | 82.8% |
| approach_reward | 55.277598 | 16.1% | 16.1% | 100.0% |
| fuel_penalty | -6.307000 | -1.8% | 1.8% | 95.3% |
| goal_progress | 1.298210 | 0.4% | 0.5% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
