# Training Feedback

## Final-policy outcome
score=-30.588550, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-62.397733, 6.664603]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress_gated | 312.791183 | 59.0% | 59.0% | 87.6% |
| fuel_penalty | -199.870000 | -37.7% | 37.7% | 99.9% |
| landing_progress | 17.344745 | 3.3% | 3.3% | 91.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
