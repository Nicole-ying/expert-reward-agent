# Training Feedback

## Final-policy outcome
score=-121.569900, len=546.450000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-234.415641, 99.438270]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_reward | 222.800000 | 69.5% | 69.5% | 3.1% |
| landing_bonus | 45.407138 | 14.2% | 14.2% | 1.9% |
| fuel_cost | -16.637000 | -5.2% | 5.2% | 96.6% |
| step_cost | -16.393500 | -5.1% | 5.1% | 100.0% |
| progress | 4.175703 | 1.3% | 3.5% | 100.0% |
| orient_cost | -3.777960 | -1.2% | 1.2% | 100.0% |
| descent_reward | 2.445011 | 0.8% | 0.8% | 95.6% |
| x_bonus | -1.887922 | -0.6% | 0.6% | 100.0% |
| slam_cost | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
