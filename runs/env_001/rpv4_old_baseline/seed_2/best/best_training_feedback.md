# Training Feedback

## Final-policy outcome
score=-10.112911, len=693.850000, terminated=18/20, truncated=2/20, reward_errors=0
score_range=[-169.623516, 205.900426]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_reward | 173.607873 | 79.7% | 79.7% | 2.6% |
| fuel_penalty | -25.070000 | -11.5% | 11.5% | 72.3% |
| time_penalty | -6.938500 | -3.2% | 3.2% | 100.0% |
| speed_penalty | -6.258816 | -2.9% | 2.9% | 100.0% |
| approach_reward | 4.244203 | 1.9% | 2.7% | 99.4% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
