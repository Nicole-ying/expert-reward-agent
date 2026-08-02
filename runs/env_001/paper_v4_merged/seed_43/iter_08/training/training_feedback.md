# Training Feedback

## Final-policy outcome
score=-124.390904, len=84.450000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-166.363526, -96.224772]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress_reward | 1.029374 | 53.5% | 56.4% | 100.0% |
| speed_penalty | -0.619512 | -32.2% | 32.2% | 0.8% |
| fuel_cost | -0.218000 | -11.3% | 11.3% | 12.9% |
| angle_penalty | -0.001613 | -0.1% | 0.1% | 0.2% |
| soft_landing_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
