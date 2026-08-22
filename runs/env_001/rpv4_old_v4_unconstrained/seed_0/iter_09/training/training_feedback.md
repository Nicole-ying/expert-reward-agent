# Training Feedback

## Final-policy outcome
score=185.785275, len=680.100000, terminated=11/20, truncated=9/20, reward_errors=0
score_range=[77.743736, 260.996126]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| proximity_reward | 289.491832 | 56.5% | 56.5% | 100.0% |
| contact_reward | 204.195000 | 39.9% | 39.9% | 54.9% |
| action_reward | -13.443000 | -2.6% | 2.6% | 72.9% |
| progress_reward | 2.547018 | 0.5% | 0.5% | 99.2% |
| vel_damping | -1.430503 | -0.3% | 0.3% | 19.5% |
| orientation_penalty | -0.809572 | -0.2% | 0.2% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
