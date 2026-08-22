# Training Feedback

## Final-policy outcome
score=146.319931, len=673.350000, terminated=14/20, truncated=6/20, reward_errors=0
score_range=[0.897306, 252.418300]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_completion | 47.039200 | 70.5% | 70.5% | 19.0% |
| distance_progress | 5.729716 | 8.6% | 12.9% | 99.5% |
| fuel_penalty | -4.571500 | -6.8% | 6.8% | 67.9% |
| stability | -3.284203 | -4.9% | 4.9% | 100.0% |
| descent_bonus | 3.276273 | 4.9% | 4.9% | 62.8% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
