# Training Feedback

## Final-policy outcome
score=-53.830120, len=545.500000, terminated=14/20, truncated=6/20, reward_errors=0
score_range=[-184.292759, 191.951763]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_bonus | 921.516514 | 94.7% | 94.7% | 3.7% |
| contact_reward | 32.380000 | 3.3% | 3.3% | 20.9% |
| action_reward | -14.110000 | -1.5% | 1.5% | 79.2% |
| progress_reward | 1.235492 | 0.1% | 0.2% | 99.9% |
| vel_damping | -1.398807 | -0.1% | 0.1% | 28.3% |
| orientation_penalty | -1.223701 | -0.1% | 0.1% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
