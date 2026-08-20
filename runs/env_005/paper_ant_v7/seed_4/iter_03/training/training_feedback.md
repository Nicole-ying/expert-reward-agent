# Training Feedback

## Final-policy outcome
score=-9.556108, len=16.350000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-49.827175, 12.839549]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_upright_gated | 9.663360 | 25.6% | 36.8% | 91.1% |
| height_gate_value | 13.371000 | 35.5% | 35.5% | 91.1% |
| rotation_penalty | -6.830848 | -18.1% | 18.1% | 90.5% |
| lateral_penalty | -3.304831 | -8.8% | 8.8% | 78.0% |
| torque_penalty | -0.306470 | -0.8% | 0.8% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
