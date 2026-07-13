# Training Feedback

## Final-policy outcome
score=1.562037, len=954.350000, terminated=3/20, truncated=17/20, reward_errors=0
score_range=[-56.841969, 157.605284]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_quality | 43.895766 | 70.9% | 82.8% | 100.0% |
| energy_penalty | -9.336000 | -15.1% | 15.1% | 97.8% |
| proximity | 1.316316 | 2.1% | 2.1% | 62.9% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
