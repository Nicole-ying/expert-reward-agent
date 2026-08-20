# Training Feedback

## Final-policy outcome
score=63.720709, len=668.050000, terminated=17/20, truncated=3/20, reward_errors=0
score_range=[-208.175480, 265.833071]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_bonus | 1075.630017 | 98.2% | 98.2% | 5.2% |
| velocity_penalty | -9.516978 | -0.9% | 0.9% | 99.0% |
| upright_penalty | -7.300626 | -0.7% | 0.7% | 100.0% |
| goal_proximity | 1.835102 | 0.2% | 0.3% | 98.8% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
