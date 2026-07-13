# Training Feedback

## Final-policy outcome
score=169.902515, len=411.650000, terminated=19/20, truncated=1/20, reward_errors=0
score_range=[-25.144351, 269.519635]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| settle_reward | 2796.576484 | 93.6% | 93.6% | 79.0% |
| speed_penalty | -131.285258 | -4.4% | 4.4% | 99.8% |
| distance_penalty | -30.616549 | -1.0% | 1.0% | 100.0% |
| progress_delta | 13.802812 | 0.5% | 0.5% | 98.5% |
| engine_penalty | -13.132500 | -0.4% | 0.4% | 63.8% |
| orientation_penalty | -1.833489 | -0.1% | 0.1% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
