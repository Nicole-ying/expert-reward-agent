# Training Feedback

## Final-policy outcome
score=-110.824742, len=68.450000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-141.582367, -92.122538]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| proximity_reward | 17.898689 | 86.8% | 90.3% | 100.0% |
| soft_landing_bonus | 1.525605 | 7.4% | 7.4% | 1.7% |
| fuel_cost | -0.465000 | -2.3% | 2.3% | 4.5% |
| velocity_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
