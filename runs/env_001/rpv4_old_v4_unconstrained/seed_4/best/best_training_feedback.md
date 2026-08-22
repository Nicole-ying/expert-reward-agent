# Training Feedback

## Final-policy outcome
score=-5.348697, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-29.437328, 29.879625]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| proximity_warmth | 45.131376 | 45.7% | 45.7% | 100.0% |
| fuel_penalty | -27.123000 | -27.5% | 27.5% | 90.4% |
| approach_reward | 19.772798 | 20.0% | 23.2% | 100.0% |
| velocity_penalty | -1.960114 | -2.0% | 2.0% | 100.0% |
| angle_penalty | -1.670077 | -1.7% | 1.7% | 100.0% |
| boundary_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |
| landing_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
