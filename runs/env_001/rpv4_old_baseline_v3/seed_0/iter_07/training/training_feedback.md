# Training Feedback

## Final-policy outcome
score=-112.760626, len=69.800000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-212.671386, 17.552794]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| success_bonus | 76.199523 | 74.5% | 74.5% | 1.0% |
| vel_cost | -7.383710 | -7.2% | 7.2% | 100.0% |
| approach_reward | 5.596897 | 5.5% | 5.7% | 100.0% |
| contact_bonus | 4.300000 | 4.2% | 4.2% | 4.8% |
| orientation_cost | -3.679135 | -3.6% | 3.6% | 100.0% |
| landing_quality | 2.876069 | 2.8% | 2.8% | 1.4% |
| proximity | 1.864612 | 1.8% | 1.8% | 100.0% |
| fuel_cost | -0.145500 | -0.1% | 0.1% | 6.9% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 18/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
