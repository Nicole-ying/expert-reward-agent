# Training Feedback

## Final-policy outcome
score=-119.141584, len=68.400000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-144.753985, -99.758755]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_success_reward | 51.671125 | 65.5% | 65.5% | 0.5% |
| approach_reward | 11.164951 | 14.1% | 14.6% | 100.0% |
| speed_penalty | -9.327596 | -11.8% | 11.8% | 99.7% |
| survival_bonus | 3.420000 | 4.3% | 4.3% | 100.0% |
| angvel_penalty | -1.833980 | -2.3% | 2.3% | 99.5% |
| centering_penalty | -0.526773 | -0.7% | 0.7% | 100.0% |
| engine_penalty | -0.445000 | -0.6% | 0.6% | 6.5% |
| angle_penalty | -0.147608 | -0.2% | 0.2% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
