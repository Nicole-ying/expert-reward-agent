# Training Feedback

## Final-policy outcome
score=79.792839, len=315.750000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-63.662425, 251.356353]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_bonus | 441.808202 | 92.2% | 92.2% | 5.7% |
| landing_approach | 17.237806 | 3.6% | 3.6% | 100.0% |
| goal_proximity | -13.301012 | -2.8% | 2.8% | 100.0% |
| velocity_penalty | -5.003746 | -1.0% | 1.0% | 98.9% |
| orientation_penalty | -1.315216 | -0.3% | 0.3% | 100.0% |
| fuel_penalty | -0.686400 | -0.1% | 0.1% | 72.5% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 1/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
