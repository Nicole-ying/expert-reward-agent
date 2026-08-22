# Training Feedback

## Final-policy outcome
score=-337.626759, len=493.450000, terminated=14/20, truncated=6/20, reward_errors=0
score_range=[-1336.137185, 33.486447]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_upright_gated | 1926.567169 | 66.7% | 85.9% | 72.4% |
| height_gate_value | 352.953315 | 12.2% | 12.2% | 73.4% |
| rotation_penalty | -31.638755 | -1.1% | 1.1% | 61.5% |
| lateral_penalty | -15.056117 | -0.5% | 0.5% | 53.3% |
| torque_penalty | -7.787898 | -0.3% | 0.3% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 2/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
