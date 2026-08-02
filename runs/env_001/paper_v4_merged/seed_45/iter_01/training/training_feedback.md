# Training Feedback

## Final-policy outcome
score=128.466338, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[100.799853, 165.827959]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress_delta | 1.400337 | 91.1% | 93.8% | 100.0% |
| speed_penalty | -0.051501 | -3.4% | 3.4% | 1.0% |
| orientation_penalty | -0.043290 | -2.8% | 2.8% | 0.6% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
