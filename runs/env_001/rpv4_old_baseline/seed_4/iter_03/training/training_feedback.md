# Training Feedback

## Final-policy outcome
score=-119.219497, len=68.300000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-144.879491, -98.137190]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| speed_penalty | -146.466949 | -66.2% | 66.2% | 100.0% |
| dist_penalty | -33.259410 | -15.0% | 15.0% | 100.0% |
| soft_contact_bonus | 14.321714 | 6.5% | 6.5% | 1.4% |
| angvel_penalty | -10.193258 | -4.6% | 4.6% | 99.6% |
| base_contact_bonus | 9.500000 | 4.3% | 4.3% | 1.4% |
| survival_penalty | -6.830000 | -3.1% | 3.1% | 100.0% |
| angle_penalty | -0.496510 | -0.2% | 0.2% | 100.0% |
| engine_penalty | -0.118500 | -0.1% | 0.1% | 5.1% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
