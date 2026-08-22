# Training Feedback

## Final-policy outcome
score=-9.288031, len=15.050000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-39.624098, 3.721936]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| upright_penalty | 6.265159 | 45.3% | 45.3% | 100.0% |
| lateral_gate | 5.897204 | 42.6% | 42.6% | 100.0% |
| gated_forward | 1.671163 | 12.1% | 12.1% | 66.4% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
