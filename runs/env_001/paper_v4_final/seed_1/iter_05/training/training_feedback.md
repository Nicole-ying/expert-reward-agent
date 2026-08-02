# Training Feedback

## Final-policy outcome
score=251.574756, len=416.700000, terminated=18/20, truncated=2/20, reward_errors=0
score_range=[126.565697, 302.989528]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_incentive | 134.299809 | 98.8% | 98.8% | 45.7% |
| progress_reward | 1.349311 | 1.0% | 1.0% | 97.7% |
| angle_penalty | -0.271513 | -0.2% | 0.2% | 1.8% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
