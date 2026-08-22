# Training Feedback

## Final-policy outcome
score=-122.209889, len=68.300000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-148.794243, -101.372852]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_success_reward | 35.000000 | 63.3% | 63.3% | 0.3% |
| approach_reward | 11.152698 | 20.2% | 20.9% | 100.0% |
| speed_penalty | -3.669613 | -6.6% | 6.6% | 100.0% |
| survival_bonus | 3.415000 | 6.2% | 6.2% | 100.0% |
| angvel_penalty | -1.156958 | -2.1% | 2.1% | 99.7% |
| engine_penalty | -0.405000 | -0.7% | 0.7% | 5.9% |
| angle_penalty | -0.091154 | -0.2% | 0.2% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
