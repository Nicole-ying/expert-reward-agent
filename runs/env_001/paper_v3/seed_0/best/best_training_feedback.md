# Training Feedback

## Final-policy outcome
score=183.641379, len=436.450000, terminated=17/20, truncated=3/20, reward_errors=0
score_range=[-18.948541, 260.802076]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_proxy | 946.382964 | 82.5% | 82.5% | 100.0% |
| distance_penalty | -191.121544 | -16.7% | 16.7% | 100.0% |
| orientation_penalty | -5.758462 | -0.5% | 0.5% | 100.0% |
| velocity_penalty | -3.810631 | -0.3% | 0.3% | 99.3% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
