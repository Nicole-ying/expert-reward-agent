# Training Feedback

## Final-policy outcome
score=-112.223731, len=68.400000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-138.377532, -93.723592]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| vel_penalty | -7.450048 | -42.6% | 42.6% | 100.0% |
| step_penalty | -6.840000 | -39.1% | 39.1% | 100.0% |
| dist_reward | 1.880874 | 10.7% | 11.2% | 100.0% |
| contact_bonus | 0.700000 | 4.0% | 4.0% | 0.5% |
| angle_penalty | -0.496536 | -2.8% | 2.8% | 100.0% |
| fuel_penalty | -0.063000 | -0.4% | 0.4% | 4.6% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
