# Training Feedback

## Final-policy outcome
score=-40.472597, len=983.300000, terminated=2/20, truncated=18/20, reward_errors=0
score_range=[-152.586351, 10.424743]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| proximity_shaping | 677.191096 | 97.8% | 97.8% | 100.0% |
| contact_reward | 7.571002 | 1.1% | 1.1% | 0.2% |
| distance_progress | 3.344686 | 0.5% | 0.6% | 100.0% |
| speed_penalty | -3.163849 | -0.5% | 0.5% | 100.0% |
| angle_near | -0.430501 | -0.1% | 0.1% | 100.0% |
| angle_global | -0.085732 | -0.0% | 0.0% | 100.0% |
| angvel_near | -0.056521 | -0.0% | 0.0% | 100.0% |
| angvel_global | -0.006701 | -0.0% | 0.0% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
