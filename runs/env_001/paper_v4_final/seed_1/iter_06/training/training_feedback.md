# Training Feedback

## Final-policy outcome
score=266.371030, len=289.100000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[239.746436, 298.721438]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_incentive | 36.527932 | 96.3% | 96.3% | 15.5% |
| progress_reward | 1.379438 | 3.6% | 3.7% | 95.0% |
| angvel_penalty | -0.002546 | -0.0% | 0.0% | 0.2% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
