# Training Feedback

## Final-policy outcome
score=144.779801, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[116.610076, 177.828587]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact | 147.920002 | 95.1% | 95.1% | 74.0% |
| speed_penalty | -3.702531 | -2.4% | 2.4% | 100.0% |
| angle_penalty | -2.530166 | -1.6% | 1.6% | 100.0% |
| progress | 1.332416 | 0.9% | 0.9% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
