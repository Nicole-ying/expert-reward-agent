# Training Feedback

## Final-policy outcome
score=77.691479, len=934.450000, terminated=7/20, truncated=13/20, reward_errors=0
score_range=[-0.803812, 200.026326]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| proximity_reward | 1734.502358 | 84.5% | 84.5% | 100.0% |
| contact_reward | 253.400000 | 12.4% | 12.4% | 6.8% |
| touchdown_bonus | 20.000000 | 1.0% | 1.0% | 0.1% |
| fuel_cost | -18.298000 | -0.9% | 0.9% | 97.9% |
| speed_penalty | -14.467388 | -0.7% | 0.7% | 6.8% |
| progress_reward | 5.179396 | 0.3% | 0.3% | 99.8% |
| orientation_penalty | -4.459569 | -0.2% | 0.2% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
