# Training Feedback

## Final-policy outcome
score=-79.636804, len=968.800000, terminated=2/20, truncated=18/20, reward_errors=0
score_range=[-176.860476, 146.328239]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_bonus | 13.688782 | 81.6% | 81.6% | 0.6% |
| progress_reward | 1.225439 | 7.3% | 14.2% | 100.0% |
| orientation_penalty | -0.695577 | -4.1% | 4.1% | 100.0% |
| velocity_penalty | -0.008640 | -0.1% | 0.1% | 0.2% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
