# Training Feedback

## Final-policy outcome
score=-101.892341, len=68.700000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-121.144123, -73.274144]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_vel_penalty | -2.747915 | -56.7% | 56.7% | 100.0% |
| progress | 1.119153 | 23.1% | 23.9% | 100.0% |
| stable_bonus | 0.471253 | 9.7% | 9.7% | 1.9% |
| upright_penalty | -0.468024 | -9.7% | 9.7% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
