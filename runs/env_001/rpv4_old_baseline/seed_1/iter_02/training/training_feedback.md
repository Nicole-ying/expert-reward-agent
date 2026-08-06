# Training Feedback

## Final-policy outcome
score=-114.198996, len=68.350000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-132.663129, -102.514675]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_bonus | 122.365160 | 37.0% | 37.0% | 1.7% |
| vel_penalty | -91.219976 | -27.5% | 27.5% | 100.0% |
| unbalanced_penalty | -47.500000 | -14.3% | 14.3% | 1.4% |
| approach_reward | 33.699102 | 10.2% | 10.5% | 100.0% |
| stability_penalty | -34.799391 | -10.5% | 10.5% | 100.0% |
| thrust_cost | -0.394500 | -0.1% | 0.1% | 19.2% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
