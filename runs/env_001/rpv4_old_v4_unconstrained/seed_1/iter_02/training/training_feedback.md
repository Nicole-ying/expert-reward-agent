# Training Feedback

## Final-policy outcome
score=-125.317262, len=68.350000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-160.183067, -64.030748]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| speed_penalty | -11.353440 | -46.3% | 46.3% | 100.0% |
| proximity_progress | 5.560546 | 22.7% | 23.5% | 100.0% |
| altitude_reward | 2.422990 | 9.9% | 9.9% | 100.0% |
| attitude_penalty | -2.149320 | -8.8% | 8.8% | 100.0% |
| step_penalty | -1.367000 | -5.6% | 5.6% | 100.0% |
| contact_reward | 1.199342 | 4.9% | 4.9% | 3.7% |
| fuel_penalty | -0.287500 | -1.2% | 1.2% | 1.7% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
