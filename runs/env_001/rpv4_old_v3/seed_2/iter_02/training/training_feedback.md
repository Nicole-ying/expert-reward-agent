# Training Feedback

## Final-policy outcome
score=-119.506414, len=68.300000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-142.824994, -96.557485]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress_shaping | 5.583480 | 50.5% | 52.3% | 100.0% |
| success_landing_bonus | 5.000000 | 45.2% | 45.2% | 0.1% |
| angle_penalty | -0.204852 | -1.9% | 1.9% | 100.0% |
| action_cost | -0.071000 | -0.6% | 0.6% | 5.2% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
