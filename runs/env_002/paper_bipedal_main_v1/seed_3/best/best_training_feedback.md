# Training Feedback

## Final-policy outcome
score=311.123548, len=1051.800000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[309.794585, 313.000250]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 504.384340 | 87.4% | 87.4% | 100.0% |
| energy_penalty | -64.113008 | -11.1% | 11.1% | 100.0% |
| angle_penalty | -7.190176 | -1.2% | 1.2% | 100.0% |
| vert_penalty | -1.218929 | -0.2% | 0.2% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
