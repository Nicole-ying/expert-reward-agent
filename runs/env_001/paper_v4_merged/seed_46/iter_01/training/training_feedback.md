# Training Feedback

## Final-policy outcome
score=236.610458, len=419.850000, terminated=17/20, truncated=3/20, reward_errors=0
score_range=[64.846763, 310.187661]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| terminal_success_bonus | 40.220000 | 96.3% | 96.3% | 47.9% |
| goal_proximity_progress | 1.326154 | 3.2% | 3.4% | 97.8% |
| orientation_penalty | -0.099911 | -0.2% | 0.2% | 2.3% |
| landing_gentleness_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
