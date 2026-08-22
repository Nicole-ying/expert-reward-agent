# Training Feedback

## Final-policy outcome
score=-240.556906, len=244.400000, terminated=18/20, truncated=2/20, reward_errors=0
score_range=[-1133.041330, -2.945546]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| health_gate | 214.880584 | 65.8% | 65.8% | 88.1% |
| forward_gated | 88.621857 | 27.1% | 27.1% | 52.5% |
| upright_penalty | -22.195416 | -6.8% | 6.8% | 12.0% |
| z_velocity_penalty | -0.731142 | -0.2% | 0.2% | 3.6% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 6/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
