# Training Feedback

## Final-policy outcome
score=-119.624725, len=68.500000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-151.335058, -24.178362]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| touchdown_bonus | 32.500000 | 54.1% | 54.1% | 0.9% |
| contact_reward | 17.000000 | 28.3% | 28.3% | 1.2% |
| speed_penalty | -3.649612 | -6.1% | 6.1% | 1.2% |
| proximity_reward | 2.941518 | 4.9% | 5.1% | 100.0% |
| progress_reward | 2.230130 | 3.7% | 3.8% | 100.0% |
| orientation_penalty | -1.527330 | -2.5% | 2.5% | 100.0% |
| fuel_cost | -0.047000 | -0.1% | 0.1% | 3.4% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 19/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
