# Training Feedback

## Final-policy outcome
score=128.439068, len=573.600000, terminated=12/20, truncated=8/20, reward_errors=0
score_range=[-35.507392, 266.659277]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| soft_landing_bonus | 535.413352 | 91.6% | 91.6% | 5.8% |
| proximity_progress | 32.604367 | 5.6% | 7.6% | 99.3% |
| attitude_penalty | -4.378923 | -0.7% | 0.7% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
