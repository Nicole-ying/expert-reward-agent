# Training Feedback

## Final-policy outcome
score=-125.711857, len=68.400000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-151.335058, -105.020073]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| success_bonus | 30.000000 | 52.2% | 52.2% | 0.2% |
| ground_danger_penalty | -15.165073 | -26.4% | 26.4% | 16.4% |
| shaping | 6.220238 | 10.8% | 11.6% | 100.0% |
| survival_bonus | 3.420000 | 6.0% | 6.0% | 100.0% |
| angle_penalty | -1.176939 | -2.0% | 2.0% | 23.1% |
| angvel_penalty | -0.497319 | -0.9% | 0.9% | 1.1% |
| contact_continuous | 0.417198 | 0.7% | 0.7% | 0.8% |
| fuel_penalty | -0.122500 | -0.2% | 0.2% | 3.6% |
| crash_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
