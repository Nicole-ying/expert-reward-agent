# Training Feedback

## Final-policy outcome
score=-57.471340, len=166.900000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-88.840113, -40.010828]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| gated_forward_speed | 80.118276 | 97.7% | 97.7% | 97.7% |
| posture_hinge_penalty | -1.879948 | -2.3% | 2.3% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 5/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
