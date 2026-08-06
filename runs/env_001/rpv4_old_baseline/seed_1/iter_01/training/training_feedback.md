# Training Feedback

## Final-policy outcome
score=-84.023947, len=69.700000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-117.723028, -43.533288]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| approach_reward | 111.137816 | 62.4% | 64.9% | 100.0% |
| landing_bonus | 52.903947 | 29.7% | 29.7% | 1.8% |
| stability_penalty | -9.133207 | -5.1% | 5.1% | 100.0% |
| thrust_cost | -0.462000 | -0.3% | 0.3% | 22.1% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 19/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
