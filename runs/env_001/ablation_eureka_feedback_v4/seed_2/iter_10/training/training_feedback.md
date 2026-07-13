# Training Feedback

## Final-policy outcome
score=-111.451519, len=68.450000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-141.582367, -94.021847]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| proximity_reward | 17.817515 | 71.0% | 73.9% | 100.0% |
| vertical_descent_penalty | -4.010448 | -16.0% | 16.0% | 74.2% |
| soft_landing_bonus | 2.130669 | 8.5% | 8.5% | 1.7% |
| fuel_cost | -0.405000 | -1.6% | 1.6% | 3.9% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
