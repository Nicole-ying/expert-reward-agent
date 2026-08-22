# Training Feedback

## Final-policy outcome
score=-15.711348, len=9.950000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-47.453503, -1.111387]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| lateral_penalty | 2.048561 | 47.5% | 47.5% | 100.0% |
| gated_forward | 2.017811 | 46.8% | 46.8% | 55.8% |
| upright_penalty | 0.243299 | 5.6% | 5.6% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
