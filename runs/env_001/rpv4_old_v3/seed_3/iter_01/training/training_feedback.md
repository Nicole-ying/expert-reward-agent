# Training Feedback

## Final-policy outcome
score=-118.279831, len=68.350000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-145.893547, -96.378786]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| approach_progress | -66.525431 | -92.1% | 92.1% | 100.0% |
| velocity_damping | -5.066365 | -7.0% | 7.0% | 100.0% |
| upright_stabilization | -0.548201 | -0.8% | 0.8% | 100.0% |
| touchdown_proxy | 0.110128 | 0.2% | 0.2% | 0.1% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
