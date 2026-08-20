# Training Feedback

## Final-policy outcome
score=149.813220, len=973.800000, terminated=1/20, truncated=19/20, reward_errors=0
score_range=[115.303280, 257.522463]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_incentive | 334.482977 | 99.4% | 99.4% | 100.0% |
| progress_reward | 1.388799 | 0.4% | 0.4% | 100.0% |
| angle_penalty | -0.625196 | -0.2% | 0.2% | 1.7% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
