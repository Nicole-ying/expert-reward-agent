# Training Feedback

## Final-policy outcome
score=-205.036446, len=785.850000, terminated=15/20, truncated=5/20, reward_errors=0
score_range=[-297.060305, -125.518562]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| height_near | 494.329784 | 66.7% | 66.7% | 100.0% |
| alignment | 231.122867 | 31.2% | 31.2% | 100.0% |
| lat_penalty | -8.057471 | -1.1% | 1.1% | 16.3% |
| still_bonus | 2.736286 | 0.4% | 0.4% | 24.0% |
| att_penalty | -2.497858 | -0.3% | 0.3% | 15.7% |
| angvel_penalty | -1.172042 | -0.2% | 0.2% | 15.7% |
| descend_vel | 1.140845 | 0.2% | 0.2% | 63.7% |
| progress_y | 0.051331 | 0.0% | 0.0% | 63.7% |
| contact_reward | 0.000000 | 0.0% | 0.0% | 0.0% |
| down_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
