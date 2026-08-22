# Training Feedback

## Final-policy outcome
score=-55.663360, len=278.150000, terminated=19/20, truncated=1/20, reward_errors=0
score_range=[-124.719580, 33.194695]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| vertical_gate | 269.750141 | 33.6% | 33.6% | 100.0% |
| health_gate | 234.835417 | 29.2% | 29.2% | 100.0% |
| angle_gate | 211.177576 | 26.3% | 26.3% | 98.8% |
| progress_raw | 83.467634 | 10.4% | 10.4% | 99.0% |
| falling_risk_penalty | 4.250245 | 0.5% | 0.5% | 48.4% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 6/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
