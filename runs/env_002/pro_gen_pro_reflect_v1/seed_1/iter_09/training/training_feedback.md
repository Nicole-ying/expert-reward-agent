# Training Feedback

## Final-policy outcome
score=251.830146, len=896.400000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-16.453449, 312.972373]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| velocity_gated | 443.867441 | 91.6% | 91.7% | 100.0% |
| action_penalty | -23.021312 | -4.8% | 4.8% | 100.0% |
| gait_bonus | 17.377500 | 3.6% | 3.6% | 64.6% |
| vertical_penalty | -0.047398 | -0.0% | 0.0% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
