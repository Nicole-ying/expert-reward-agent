# Training Feedback

## Final-policy outcome
score=226.436359, len=599.100000, terminated=12/20, truncated=8/20, reward_errors=0
score_range=[129.817242, 315.811652]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_proxy | 101.160778 | 98.5% | 98.5% | 58.3% |
| progress | 1.384166 | 1.3% | 1.4% | 98.6% |
| angle_penalty | -0.070187 | -0.1% | 0.1% | 2.2% |
| ang_vel_penalty | -0.000054 | -0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
