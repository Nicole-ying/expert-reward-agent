# Training Feedback

## Final-policy outcome
score=-115.060376, len=68.450000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-140.542772, -97.885900]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| speed_penalty | -36.554002 | -39.9% | 39.9% | 100.0% |
| approach_reward | 22.389281 | 24.4% | 25.3% | 100.0% |
| contact_reward | 15.192044 | 16.6% | 16.6% | 0.7% |
| angvel_penalty | -8.864465 | -9.7% | 9.7% | 99.6% |
| survival_penalty | -6.845000 | -7.5% | 7.5% | 100.0% |
| angle_penalty | -0.580428 | -0.6% | 0.6% | 100.0% |
| engine_penalty | -0.387000 | -0.4% | 0.4% | 11.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
