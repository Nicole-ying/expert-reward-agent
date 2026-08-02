# Training Feedback

## Final-policy outcome
score=-111.409504, len=68.450000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-130.746612, -92.593304]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress_delta | 5.597073 | 49.9% | 51.7% | 100.0% |
| speed_penalty | -4.616375 | -41.1% | 41.1% | 90.2% |
| completion_proxy | 0.442435 | 3.9% | 3.9% | 0.7% |
| angvel_penalty | -0.198741 | -1.8% | 1.8% | 0.9% |
| engine_penalty | -0.165000 | -1.5% | 1.5% | 4.8% |
| angle_penalty | -0.001784 | -0.0% | 0.0% | 0.2% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
