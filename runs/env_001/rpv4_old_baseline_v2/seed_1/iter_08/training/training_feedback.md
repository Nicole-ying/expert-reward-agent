# Training Feedback

## Final-policy outcome
score=-16.940151, len=434.650000, terminated=16/20, truncated=4/20, reward_errors=0
score_range=[-172.726719, 283.226261]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_reward | 419.600000 | 72.4% | 72.4% | 10.4% |
| distance_penalty | -145.014200 | -25.0% | 25.0% | 100.0% |
| success_bonus | 7.400000 | 1.3% | 1.3% | 0.9% |
| vel_penalty | -6.076252 | -1.0% | 1.0% | 99.5% |
| angle_penalty | -1.412958 | -0.2% | 0.2% | 100.0% |
| angvel_penalty | -0.128159 | -0.0% | 0.0% | 98.9% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
