# Training Feedback

## Final-policy outcome
score=-63.659559, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-93.268824, -22.954316]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| distance_progress | 1.516268 | 57.3% | 95.1% | 100.0% |
| angle_penalty | -0.118193 | -4.5% | 4.5% | 100.0% |
| angvel_penalty | -0.012110 | -0.5% | 0.5% | 100.0% |
| contact_reward | 0.000000 | 0.0% | 0.0% | 0.0% |
| landing_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
