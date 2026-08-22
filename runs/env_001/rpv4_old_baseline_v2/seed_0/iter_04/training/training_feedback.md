# Training Feedback

## Final-policy outcome
score=-114.951866, len=68.400000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-139.287477, -95.462869]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_change | 2.100201 | 70.2% | 70.6% | 3.1% |
| progress_reward | 0.540807 | 18.1% | 20.3% | 100.0% |
| vel_change | -0.002102 | -0.1% | 5.4% | 100.0% |
| fuel_penalty | -0.089000 | -3.0% | 3.0% | 6.5% |
| ang_change | -0.006531 | -0.2% | 0.7% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
