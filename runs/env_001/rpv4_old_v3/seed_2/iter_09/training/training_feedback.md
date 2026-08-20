# Training Feedback

## Final-policy outcome
score=145.168065, len=718.800000, terminated=9/20, truncated=11/20, reward_errors=0
score_range=[33.531595, 288.928579]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| success_landing_bonus | 5330.000000 | 99.8% | 99.8% | 7.4% |
| fuel_penalty | -4.267500 | -0.1% | 0.1% | 59.4% |
| progress | 2.637606 | 0.0% | 0.1% | 99.2% |
| vel_damping | -2.021547 | -0.0% | 0.0% | 99.2% |
| angle_stab | -0.748251 | -0.0% | 0.0% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
