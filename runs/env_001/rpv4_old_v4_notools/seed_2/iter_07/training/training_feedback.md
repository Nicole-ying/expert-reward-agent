# Training Feedback

## Final-policy outcome
score=83.409540, len=760.400000, terminated=9/20, truncated=11/20, reward_errors=0
score_range=[-133.684066, 219.487212]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_bonus | 400.000000 | 67.0% | 67.0% | 9.9% |
| stable_contact_bonus | 181.200000 | 30.4% | 30.4% | 47.7% |
| progress | 8.986431 | 1.5% | 2.0% | 99.8% |
| orientation_penalty | -2.361863 | -0.4% | 0.4% | 100.0% |
| velocity_penalty | -1.117757 | -0.2% | 0.2% | 99.6% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
