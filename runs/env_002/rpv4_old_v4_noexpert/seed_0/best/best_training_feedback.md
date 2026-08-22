# Training Feedback

## Final-policy outcome
score=299.433589, len=1322.950000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[296.908606, 301.775437]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_progress | 593.625040 | 91.0% | 91.0% | 100.0% |
| energy_penalty | -29.363293 | -4.5% | 4.5% | 100.0% |
| torso_stability | -28.663635 | -4.4% | 4.4% | 100.0% |
| vertical_velocity_penalty | -0.388578 | -0.1% | 0.1% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
