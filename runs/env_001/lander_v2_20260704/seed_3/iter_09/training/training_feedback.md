# Training Feedback

## Final-policy outcome
score=-33.974125, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-68.988164, 4.597475]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| proximity_quality | 809.850599 | 99.8% | 99.8% | 100.0% |
| approach_reward | 1.065147 | 0.1% | 0.2% | 100.0% |
| stability_penalty | -0.612114 | -0.1% | 0.1% | 100.0% |
| contact_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
