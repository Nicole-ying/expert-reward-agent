# Training Feedback

## Final-policy outcome
score=-110.442116, len=68.450000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-130.746612, -95.059093]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| proximity_reward | 17.906316 | 81.9% | 85.3% | 100.0% |
| soft_landing_bonus | 2.137287 | 9.8% | 9.8% | 1.7% |
| velocity_penalty | -0.700594 | -3.2% | 3.2% | 99.6% |
| fuel_cost | -0.382500 | -1.7% | 1.7% | 3.7% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
