# Training Feedback

## Final-policy outcome
score=-117.693633, len=68.300000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-141.465656, -95.448355]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| vel_penalty | -23.024253 | -60.0% | 60.0% | 46.8% |
| dist_reward | 8.937873 | 23.3% | 24.1% | 100.0% |
| contact_bonus | 3.375000 | 8.8% | 8.8% | 2.9% |
| step_penalty | -1.366000 | -3.6% | 3.6% | 100.0% |
| angle_penalty | -1.299091 | -3.4% | 3.4% | 100.0% |
| fuel_penalty | -0.033000 | -0.1% | 0.1% | 4.8% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
