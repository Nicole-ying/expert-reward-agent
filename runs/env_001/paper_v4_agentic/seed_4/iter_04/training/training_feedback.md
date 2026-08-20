# Training Feedback

## Final-policy outcome
score=-4.761621, len=999.450000, terminated=1/20, truncated=19/20, reward_errors=0
score_range=[-41.698261, 77.305001]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| proximity_stability | 4364.715619 | 88.6% | 88.6% | 81.6% |
| progress_gated | 361.375662 | 7.3% | 7.3% | 75.2% |
| fuel_penalty | -199.890000 | -4.1% | 4.1% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
