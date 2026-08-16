# Training Feedback

## Final-policy outcome
score=87.303733, len=942.500000, terminated=2/20, truncated=18/20, reward_errors=0
score_range=[34.736439, 262.984207]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_reward | 3007.164113 | 50.3% | 50.3% | 17.7% |
| contact_loss_penalty | -2625.000000 | -43.9% | 43.9% | 14.5% |
| slow_bonus | 271.582698 | 4.5% | 4.5% | 100.0% |
| fuel_penalty | -28.036500 | -0.5% | 0.5% | 99.2% |
| approach_reward | 20.290261 | 0.3% | 0.4% | 100.0% |
| step_penalty | -18.850000 | -0.3% | 0.3% | 100.0% |
| speed_penalty | -4.752794 | -0.1% | 0.1% | 100.0% |
| attitude_penalty | -1.683077 | -0.0% | 0.0% | 100.0% |
| descent_incentive | 0.788456 | 0.0% | 0.0% | 62.7% |
| centering_penalty | -0.439465 | -0.0% | 0.0% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
