# Training Feedback

## Final-policy outcome
score=-119.090536, len=68.450000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-141.274164, -95.259451]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_bonus | 55.000000 | 64.9% | 64.9% | 0.8% |
| speed_penalty | -27.315364 | -32.2% | 32.2% | 100.0% |
| progress_reward | 1.126019 | 1.3% | 1.4% | 100.0% |
| attitude_penalty | -1.093118 | -1.3% | 1.3% | 100.0% |
| contact_reward | 0.220000 | 0.3% | 0.3% | 1.6% |
| energy_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
