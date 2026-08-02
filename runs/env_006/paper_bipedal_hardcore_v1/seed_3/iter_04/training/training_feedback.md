# Training Feedback

## Final-policy outcome
score=-65.667744, len=148.400000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-86.360993, -44.048082]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| posture_gate | 103.250916 | 88.0% | 88.0% | 100.0% |
| progress_reward | 13.970665 | 11.9% | 11.9% | 100.0% |
| vertical_penalty | -0.107135 | -0.1% | 0.1% | 100.0% |
| air_penalty | -0.039000 | -0.0% | 0.0% | 1.8% |
| angular_penalty | -0.021229 | -0.0% | 0.0% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 13/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
