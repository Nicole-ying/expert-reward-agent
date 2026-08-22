# Training Feedback

## Final-policy outcome
score=-33.405803, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-57.684245, 4.059924]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| approach_vel | 13.902619 | 81.2% | 92.1% | 100.0% |
| near_speed_penalty | -0.771197 | -4.5% | 4.5% | 100.0% |
| progress | 0.532170 | 3.1% | 3.4% | 100.0% |
| orientation | -0.000546 | -0.0% | 0.0% | 100.0% |
| landing | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
