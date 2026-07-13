# Training Feedback

## Final-policy outcome
score=130.636384, len=647.800000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-90.281245, 201.279227]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_bonus | 1655.035671 | 90.5% | 90.5% | 2.8% |
| contact_reward | 171.200000 | 9.4% | 9.4% | 7.5% |
| distance_progress | 2.427177 | 0.1% | 0.1% | 88.5% |
| angle_penalty | -0.210202 | -0.0% | 0.0% | 100.0% |
| angvel_penalty | -0.034771 | -0.0% | 0.0% | 97.7% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
