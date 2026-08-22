# Training Feedback

## Final-policy outcome
score=13.465799, len=989.500000, terminated=3/20, truncated=17/20, reward_errors=0
score_range=[-41.158885, 171.621777]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_bonus | 614.992574 | 88.6% | 88.6% | 93.9% |
| fuel_penalty | -48.767500 | -7.0% | 7.0% | 98.6% |
| progress | 24.984360 | 3.6% | 4.3% | 99.8% |
| orientation_penalty | -0.432990 | -0.1% | 0.1% | 83.3% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
