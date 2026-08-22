# Training Feedback

## Final-policy outcome
score=-7.145291, len=838.500000, terminated=7/20, truncated=13/20, reward_errors=0
score_range=[-85.917549, 216.358157]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| vertical_guide | -308.620692 | -51.0% | 51.0% | 96.5% |
| horizontal_speed_penalty | -108.738401 | -18.0% | 18.0% | 86.6% |
| fuel_penalty | -78.789000 | -13.0% | 13.0% | 91.5% |
| success_bonus | 45.000000 | 7.4% | 7.4% | 0.1% |
| shaping | 8.508295 | 1.4% | 7.4% | 100.0% |
| angle_penalty | -15.407152 | -2.5% | 2.5% | 100.0% |
| contact_reward | 2.288187 | 0.4% | 0.4% | 0.5% |
| angvel_penalty | -1.104213 | -0.2% | 0.2% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
