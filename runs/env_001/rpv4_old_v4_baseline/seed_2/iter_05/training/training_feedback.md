# Training Feedback

## Final-policy outcome
score=170.176332, len=899.850000, terminated=3/20, truncated=17/20, reward_errors=0
score_range=[16.802296, 311.799840]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_guidance | 71.831656 | 92.6% | 92.6% | 81.0% |
| progress | 4.102001 | 5.3% | 5.6% | 99.8% |
| orientation_penalty | -1.132255 | -1.5% | 1.5% | 100.0% |
| velocity_penalty | -0.228543 | -0.3% | 0.3% | 99.7% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
