# Training Feedback

## Final-policy outcome
score=171.408028, len=781.350000, terminated=7/20, truncated=13/20, reward_errors=0
score_range=[106.483334, 281.700870]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_guidance | 252.864200 | 99.2% | 99.2% | 68.0% |
| orientation_penalty | -1.197761 | -0.5% | 0.5% | 100.0% |
| progress | 0.650697 | 0.3% | 0.3% | 99.9% |
| velocity_penalty | -0.180795 | -0.1% | 0.1% | 99.8% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
