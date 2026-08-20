# Training Feedback

## Final-policy outcome
score=71.544387, len=918.650000, terminated=7/20, truncated=13/20, reward_errors=0
score_range=[-7.434875, 193.944703]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_bonus | 2862.500000 | 51.7% | 51.7% | 6.2% |
| descent_quality | 1971.595052 | 35.6% | 35.6% | 100.0% |
| proximity | 703.618274 | 12.7% | 12.7% | 100.0% |
| velocity_penalty | -0.229481 | -0.0% | 0.0% | 99.5% |
| attitude_penalty | -0.051558 | -0.0% | 0.0% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
