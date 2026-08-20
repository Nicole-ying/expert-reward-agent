# Training Feedback

## Final-policy outcome
score=-416.103382, len=625.100000, terminated=11/20, truncated=9/20, reward_errors=0
score_range=[-1093.286726, -0.983797]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_gated | 274.210306 | 75.7% | 84.2% | 77.3% |
| lateral_penalty | -30.771728 | -8.5% | 8.5% | 76.4% |
| action_penalty | -26.586863 | -7.3% | 7.3% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 3/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
