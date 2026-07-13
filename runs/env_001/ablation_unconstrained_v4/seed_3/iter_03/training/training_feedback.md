# Training Feedback

## Final-policy outcome
score=-20.703763, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-55.161121, 16.171573]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| height_reward | 340.273689 | 44.8% | 44.8% | 100.0% |
| goal_reward | 333.030656 | 43.8% | 43.8% | 100.0% |
| engine_penalty | -86.980000 | -11.4% | 11.4% | 87.0% |
| contact_reward | 0.000000 | 0.0% | 0.0% | 0.0% |
| landing_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
