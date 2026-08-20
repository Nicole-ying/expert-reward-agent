# Training Feedback

## Final-policy outcome
score=91.816111, len=900.700000, terminated=5/20, truncated=15/20, reward_errors=0
score_range=[-8.595378, 239.534670]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_bonus | 1029.163352 | 95.1% | 95.1% | 94.4% |
| fuel_penalty | -45.035000 | -4.2% | 4.2% | 100.0% |
| progress | 6.360760 | 0.6% | 0.7% | 100.0% |
| orientation_penalty | -0.634227 | -0.1% | 0.1% | 85.2% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
