# Training Feedback

## Final-policy outcome
score=-114.278961, len=68.450000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-139.681272, -90.479014]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| success_bonus | 77.198031 | 51.8% | 51.8% | 1.2% |
| near_speed_penalty | -30.589482 | -20.5% | 20.5% | 100.0% |
| drift_penalty | -11.271814 | -7.6% | 7.6% | 100.0% |
| approach_reward | 8.945437 | 6.0% | 6.2% | 100.0% |
| vel_cost | -7.325735 | -4.9% | 4.9% | 100.0% |
| contact_bonus | 6.400000 | 4.3% | 4.3% | 3.1% |
| landing_quality | 3.609407 | 2.4% | 2.4% | 1.5% |
| orientation_cost | -3.393472 | -2.3% | 2.3% | 100.0% |
| fuel_cost | -0.106500 | -0.1% | 0.1% | 5.2% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
