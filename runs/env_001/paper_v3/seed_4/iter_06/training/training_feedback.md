# Training Feedback

## Final-policy outcome
score=4.293878, len=949.850000, terminated=5/20, truncated=15/20, reward_errors=0
score_range=[-52.570470, 136.300510]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| proximity | 694.202770 | 78.4% | 78.4% | 100.0% |
| descent_quality | 138.722873 | 15.7% | 15.7% | 100.0% |
| contact_quality | 52.608444 | 5.9% | 5.9% | 1.5% |
| velocity_penalty | -0.238645 | -0.0% | 0.0% | 99.8% |
| attitude_penalty | -0.072776 | -0.0% | 0.0% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
