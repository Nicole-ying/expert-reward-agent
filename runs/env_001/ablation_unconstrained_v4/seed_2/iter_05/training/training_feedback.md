# Training Feedback

## Final-policy outcome
score=-69.727589, len=937.950000, terminated=5/20, truncated=15/20, reward_errors=0
score_range=[-191.368841, -7.954588]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| height_near_reward | 1654.128514 | 79.1% | 79.1% | 89.6% |
| alignment | 372.461666 | 17.8% | 17.8% | 89.6% |
| any_contact_reward | 42.500000 | 2.0% | 2.0% | 0.9% |
| y_progress | 11.883928 | 0.6% | 0.6% | 100.0% |
| progress | 4.275863 | 0.2% | 0.3% | 100.0% |
| horizontal_penalty | -2.482981 | -0.1% | 0.1% | 100.0% |
| att_penalty | -0.051837 | -0.0% | 0.0% | 4.6% |
| vel_penalty | -0.008214 | -0.0% | 0.0% | 1.2% |
| stable_contact_reward | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
