# Training Feedback

## Final-policy outcome
score=29.828681, len=844.750000, terminated=10/20, truncated=10/20, reward_errors=0
score_range=[-102.784991, 166.623024]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 288.207456 | 64.4% | 64.4% | 100.0% |
| fall_risk_penalty | -158.047590 | -35.3% | 35.3% | 7.4% |
| stability_penalty | -1.278112 | -0.3% | 0.3% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 10/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
