# Training Feedback

## Final-policy outcome
score=297.263114, len=997.800000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[130.082472, 307.378437]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| gated_forward_reward | 401.342414 | 96.5% | 96.5% | 99.2% |
| energy_penalty | -14.531943 | -3.5% | 3.5% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
