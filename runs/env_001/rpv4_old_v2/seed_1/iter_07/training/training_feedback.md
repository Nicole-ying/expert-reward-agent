# Training Feedback

## Final-policy outcome
score=-133.267334, len=656.000000, terminated=18/20, truncated=2/20, reward_errors=0
score_range=[-214.122000, -3.501409]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_bonus | 561.361123 | 94.6% | 94.6% | 1.1% |
| fuel_penalty | -22.590000 | -3.8% | 3.8% | 68.9% |
| progress | 2.059187 | 0.3% | 1.4% | 100.0% |
| orientation_penalty | -1.352215 | -0.2% | 0.2% | 83.2% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
