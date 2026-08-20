# Training Feedback

## Final-policy outcome
score=-19.587337, len=78.250000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-68.523796, 18.316422]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| proximity | -153.498672 | -92.7% | 92.7% | 100.0% |
| velocity_penalty | -11.140379 | -6.7% | 6.7% | 99.9% |
| orientation_penalty | -0.611322 | -0.4% | 0.4% | 100.0% |
| contact_bonus | 0.345483 | 0.2% | 0.2% | 2.7% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 3/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
