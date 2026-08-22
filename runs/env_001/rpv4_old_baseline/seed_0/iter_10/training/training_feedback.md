# Training Feedback

## Final-policy outcome
score=-119.398192, len=68.300000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-144.863067, -98.019399]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_bonus | 257.342415 | 83.7% | 85.0% | 2.9% |
| speed_penalty | -27.908206 | -9.1% | 9.1% | 99.7% |
| proximity_reward | 5.581956 | 1.8% | 1.9% | 100.0% |
| x_penalty | -5.358649 | -1.7% | 1.7% | 100.0% |
| time_penalty | -3.415000 | -1.1% | 1.1% | 100.0% |
| angle_penalty | -3.135637 | -1.0% | 1.0% | 100.0% |
| engine_penalty | -0.562500 | -0.2% | 0.2% | 5.5% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
