# Training Feedback

## Final-policy outcome
score=-100.821003, len=71.500000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-202.913329, 25.948439]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| approach_reward | 13.531166 | 53.8% | 55.9% | 100.0% |
| speed_penalty | -6.092345 | -24.2% | 24.2% | 100.0% |
| attitude_penalty | -2.878651 | -11.5% | 11.5% | 100.0% |
| contact_gain_reward | 0.753144 | 3.0% | 3.0% | 2.2% |
| contact_loss_penalty | -0.500000 | -2.0% | 2.0% | 0.3% |
| descent_incentive | 0.431431 | 1.7% | 1.7% | 94.5% |
| centering_penalty | -0.263810 | -1.0% | 1.0% | 100.0% |
| fuel_penalty | -0.169000 | -0.7% | 0.7% | 11.8% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 18/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
