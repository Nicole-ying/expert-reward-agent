# Training Feedback

## Final-policy outcome
score=-93.264022, len=70.400000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-141.385890, -46.322790]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| approach_reward | 13.224344 | 67.9% | 70.8% | 100.0% |
| attitude_penalty | -1.836411 | -9.4% | 9.4% | 100.0% |
| speed_penalty | -1.173432 | -6.0% | 6.0% | 100.0% |
| slow_bonus | 0.947411 | 4.9% | 4.9% | 100.0% |
| contact_gain_reward | 0.606460 | 3.1% | 3.1% | 2.3% |
| descent_incentive | 0.429514 | 2.2% | 2.2% | 94.5% |
| centering_penalty | -0.248010 | -1.3% | 1.3% | 100.0% |
| fuel_penalty | -0.245000 | -1.3% | 1.3% | 17.4% |
| contact_loss_penalty | -0.200000 | -1.0% | 1.0% | 0.1% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 17/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
