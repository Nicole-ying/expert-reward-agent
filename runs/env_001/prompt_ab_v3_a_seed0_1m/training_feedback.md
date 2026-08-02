# Training Feedback

## Final-policy outcome
score=-121.544410, len=68.400000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-147.382577, -105.977095]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| stable_landing | -2.750000 | -39.6% | 82.8% | 1.5% |
| progress | 1.118566 | 16.1% | 16.7% | 100.0% |
| thrust_cost | -0.036000 | -0.5% | 0.5% | 5.3% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
