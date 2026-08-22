# Training Feedback

## Final-policy outcome
score=180.657980, len=847.700000, terminated=6/20, truncated=14/20, reward_errors=0
score_range=[117.539822, 294.385500]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_incentive | 244.606519 | 99.2% | 99.2% | 100.0% |
| progress_reward | 1.368117 | 0.6% | 0.6% | 99.5% |
| angle_penalty | -0.592190 | -0.2% | 0.2% | 1.5% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
