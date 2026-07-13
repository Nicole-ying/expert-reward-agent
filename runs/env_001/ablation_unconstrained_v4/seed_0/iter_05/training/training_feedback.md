# Training Feedback

## Final-policy outcome
score=-117.746710, len=68.350000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-143.386473, -95.658444]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| distance_penalty | -19.952803 | -47.1% | 47.1% | 100.0% |
| progress_delta | 13.414139 | 31.7% | 32.8% | 100.0% |
| velocity_damping | -6.734450 | -15.9% | 15.9% | 100.0% |
| landing_reward | 1.430447 | 3.4% | 3.4% | 0.4% |
| orientation_penalty | -0.227389 | -0.5% | 0.5% | 100.0% |
| engine_penalty | -0.106500 | -0.3% | 0.3% | 5.2% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
