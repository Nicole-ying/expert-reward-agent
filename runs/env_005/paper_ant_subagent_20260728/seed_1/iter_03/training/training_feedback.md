# Training Feedback

## Final-policy outcome
score=-101.968889, len=319.600000, terminated=16/20, truncated=4/20, reward_errors=0
score_range=[-499.304153, 35.789369]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| upright_penalty | -141.076971 | -34.2% | 34.2% | 59.4% |
| forward_gated | 122.707676 | 29.7% | 33.8% | 99.9% |
| health_gate | 130.632417 | 31.7% | 31.7% | 41.1% |
| z_velocity_penalty | -1.593179 | -0.4% | 0.4% | 4.9% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
