# Training Feedback

## Final-policy outcome
score=-285.172350, len=86.150000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-469.550084, -161.138554]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress | 0.848496 | 54.8% | 60.7% | 100.0% |
| fuel_cost | -0.398000 | -25.7% | 25.7% | 23.1% |
| soft_landing | -0.209864 | -13.6% | 13.6% | 1.9% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
