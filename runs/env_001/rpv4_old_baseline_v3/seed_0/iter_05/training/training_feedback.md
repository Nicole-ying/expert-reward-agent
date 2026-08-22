# Training Feedback

## Final-policy outcome
score=-48.745400, len=101.800000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-133.405059, 33.235863]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| success_bonus | 63.196637 | 28.6% | 28.6% | 0.9% |
| speed_shaping | -58.994000 | -26.7% | 26.7% | 100.0% |
| contact_bonus | 40.100000 | 18.1% | 18.1% | 11.3% |
| landing_quality | 28.104392 | 12.7% | 12.7% | 8.3% |
| drift_penalty | -12.456011 | -5.6% | 5.6% | 100.0% |
| orientation_cost | -10.651687 | -4.8% | 4.8% | 100.0% |
| approach_reward | 5.491142 | 2.5% | 2.6% | 100.0% |
| fuel_cost | -1.858000 | -0.8% | 0.8% | 45.6% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 8/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
