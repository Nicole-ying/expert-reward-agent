# Training Feedback

## Final-policy outcome
score=-113.587753, len=105.250000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-154.779871, -73.148435]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress_gate_reward | 0.904320 | 74.2% | 74.2% | 93.3% |
| energy_penalty | -0.314000 | -25.8% | 25.8% | 29.8% |
| terminal_velocity_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 18/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
