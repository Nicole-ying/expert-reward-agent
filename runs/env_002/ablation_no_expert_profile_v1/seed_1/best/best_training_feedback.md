# Training Feedback

## Final-policy outcome
score=258.872927, len=968.700000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[71.216349, 297.992997]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_velocity | 473.963326 | 96.6% | 96.7% | 100.0% |
| upright_penalty | -16.338337 | -3.3% | 3.3% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
