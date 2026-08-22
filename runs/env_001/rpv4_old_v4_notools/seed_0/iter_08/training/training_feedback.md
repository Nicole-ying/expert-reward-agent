# Training Feedback

## Final-policy outcome
score=-477.404463, len=112.900000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-591.432964, -352.794057]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress | 1.849271 | 62.0% | 100.0% | 100.0% |
| contact_quality | 0.000000 | 0.0% | 0.0% | 0.0% |
| landing_gate_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 18/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
