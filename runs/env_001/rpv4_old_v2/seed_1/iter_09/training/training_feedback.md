# Training Feedback

## Final-policy outcome
score=-98.538447, len=69.700000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-140.646332, -25.547924]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress | 4.620166 | 50.5% | 52.8% | 100.0% |
| orientation_penalty | -3.297694 | -36.1% | 36.1% | 22.7% |
| contact_bonus | 0.573659 | 6.3% | 6.3% | 1.1% |
| fuel_penalty | -0.445000 | -4.9% | 4.9% | 12.8% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 17/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
