# Training Feedback

## Final-policy outcome
score=-124.583385, len=522.750000, terminated=18/20, truncated=2/20, reward_errors=0
score_range=[-251.616878, -58.025817]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_reward | 34.500000 | 39.5% | 39.5% | 0.5% |
| touchdown_bonus | 25.000000 | 28.6% | 28.6% | 0.0% |
| fuel_cost | -13.869000 | -15.9% | 15.9% | 88.4% |
| orientation_penalty | -9.568153 | -11.0% | 11.0% | 100.0% |
| progress_reward | 0.987879 | 1.1% | 4.7% | 100.0% |
| speed_penalty | -0.247500 | -0.3% | 0.3% | 0.1% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
