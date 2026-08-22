# Training Feedback

## Final-policy outcome
score=53.731200, len=910.900000, terminated=8/20, truncated=12/20, reward_errors=0
score_range=[-39.516233, 169.202575]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress_reward | 699.727172 | 82.3% | 82.3% | 100.0% |
| landing_speed_penalty | -65.773347 | -7.7% | 7.7% | 100.0% |
| fuel_penalty | -44.342500 | -5.2% | 5.2% | 97.4% |
| contact_reward | 39.595445 | 4.7% | 4.7% | 2.6% |
| settlement_penalty | -1.162918 | -0.1% | 0.1% | 2.6% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
