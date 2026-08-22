# Training Feedback

## Final-policy outcome
score=-10.305339, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-39.145883, 27.549397]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| proximity | 37.139062 | 37.9% | 37.9% | 100.0% |
| main_decel_bonus | 22.937084 | 23.4% | 23.4% | 53.1% |
| fuel_cost | -19.137000 | -19.5% | 19.5% | 63.8% |
| vel_cost | -10.869547 | -11.1% | 11.1% | 100.0% |
| approach_reward | 5.907605 | 6.0% | 7.0% | 100.0% |
| orientation_cost | -0.963790 | -1.0% | 1.0% | 100.0% |
| contact_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |
| crash_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |
| landing_quality | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
