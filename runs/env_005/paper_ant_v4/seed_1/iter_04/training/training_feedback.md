# Training Feedback

## Final-policy outcome
score=86.944599, len=950.600000, terminated=1/20, truncated=19/20, reward_errors=0
score_range=[-287.983032, 404.822037]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_progress | 882.882622 | 20.2% | 78.6% | 90.1% |
| upright_stability | 932.448996 | 21.3% | 21.3% | 100.0% |
| lateral_drift_penalty | -2.030486 | -0.0% | 0.0% | 51.8% |
| height_penalty | -0.008098 | -0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
