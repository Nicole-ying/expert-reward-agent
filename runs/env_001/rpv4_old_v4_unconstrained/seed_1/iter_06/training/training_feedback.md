# Training Feedback

## Final-policy outcome
score=-40.995321, len=976.650000, terminated=1/20, truncated=19/20, reward_errors=0
score_range=[-86.074082, -2.576422]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| soft_landing | 306.664896 | 87.3% | 87.3% | 100.0% |
| fuel_penalty | -19.235000 | -5.5% | 5.5% | 98.5% |
| approach_reward | 12.357417 | 3.5% | 5.1% | 100.0% |
| centering_penalty | -6.084178 | -1.7% | 1.7% | 100.0% |
| attitude_penalty | -1.058117 | -0.3% | 0.3% | 100.0% |
| descent_incentive | 0.414252 | 0.1% | 0.1% | 67.5% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
