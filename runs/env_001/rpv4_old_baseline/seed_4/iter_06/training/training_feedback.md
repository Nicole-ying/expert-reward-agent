# Training Feedback

## Final-policy outcome
score=-123.873980, len=68.300000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-156.391859, -75.292707]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_success_reward | 40.000000 | 69.7% | 69.7% | 0.3% |
| approach_reward | 5.568204 | 9.7% | 10.0% | 100.0% |
| speed_penalty | -3.671942 | -6.4% | 6.4% | 100.0% |
| survival_bonus | 3.415000 | 6.0% | 6.0% | 100.0% |
| contact_bonus | 3.200000 | 5.6% | 5.6% | 3.4% |
| angvel_penalty | -0.940885 | -1.6% | 1.6% | 99.6% |
| engine_penalty | -0.220000 | -0.4% | 0.4% | 3.2% |
| angle_penalty | -0.154154 | -0.3% | 0.3% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
