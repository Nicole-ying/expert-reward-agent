# Training Feedback

## Final-policy outcome
score=-110.570500, len=68.450000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-124.839570, -95.059083]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| speed_penalty | -18.338892 | -51.4% | 51.4% | 100.0% |
| goal_proximity | 10.842741 | 30.4% | 34.1% | 100.0% |
| landing_contact | 3.500000 | 9.8% | 9.8% | 0.5% |
| attitude_constraint | -1.530701 | -4.3% | 4.3% | 100.0% |
| fuel_efficiency | -0.137500 | -0.4% | 0.4% | 4.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
