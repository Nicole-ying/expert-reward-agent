# Training Feedback

## Final-policy outcome
score=150.945727, len=796.250000, terminated=16/20, truncated=4/20, reward_errors=0
score_range=[-18.632940, 232.046951]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_quality | 35.259358 | 93.9% | 93.9% | 4.5% |
| progress | 1.386574 | 3.7% | 4.2% | 98.5% |
| orientation_penalty | -0.721441 | -1.9% | 1.9% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
