# Training Feedback

## Final-policy outcome
score=26.451246, len=206.500000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[0.550268, 50.007715]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress | 2.666861 | 45.6% | 47.5% | 100.0% |
| vel_damping | -1.576654 | -27.0% | 27.0% | 99.9% |
| fuel_penalty | -1.231500 | -21.1% | 21.1% | 59.6% |
| angle_stab | -0.263480 | -4.5% | 4.5% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
