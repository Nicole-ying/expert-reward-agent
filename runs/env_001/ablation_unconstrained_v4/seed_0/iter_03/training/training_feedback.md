# Training Feedback

## Final-policy outcome
score=-121.198260, len=68.400000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-145.325878, -96.939078]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| distance_penalty | -33.301118 | -52.2% | 52.2% | 100.0% |
| velocity_damping | -20.208528 | -31.7% | 31.7% | 100.0% |
| settle_delta | 5.716837 | 9.0% | 9.5% | 100.0% |
| progress_delta | 3.344923 | 5.2% | 5.4% | 100.0% |
| orientation_penalty | -0.636268 | -1.0% | 1.0% | 100.0% |
| engine_penalty | -0.182500 | -0.3% | 0.3% | 5.3% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
