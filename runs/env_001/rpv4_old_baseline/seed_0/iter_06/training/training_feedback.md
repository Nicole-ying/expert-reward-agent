# Training Feedback

## Final-policy outcome
score=-28.310436, len=864.250000, terminated=3/20, truncated=17/20, reward_errors=0
score_range=[-62.489033, 74.262232]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| proximity_bonus | 763.170112 | 47.3% | 47.3% | 90.3% |
| landing_bonus | 245.250000 | 15.2% | 15.8% | 0.4% |
| radial_reward | 117.467389 | 7.3% | 15.5% | 100.0% |
| engine_penalty | -226.890000 | -14.1% | 14.1% | 74.0% |
| angle_penalty | -53.044686 | -3.3% | 3.3% | 100.0% |
| time_penalty | -43.212500 | -2.7% | 2.7% | 100.0% |
| vel_penalty | -22.146501 | -1.4% | 1.4% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
