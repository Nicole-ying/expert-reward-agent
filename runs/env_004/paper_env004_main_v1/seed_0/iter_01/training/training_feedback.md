# Training Feedback

## Final-policy outcome
score=2663.085294, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[2640.681313, 2823.904018]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_stability_reward | 1420.921728 | 99.1% | 99.1% | 100.0% |
| stability_penalty | -13.434751 | -0.9% | 0.9% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
