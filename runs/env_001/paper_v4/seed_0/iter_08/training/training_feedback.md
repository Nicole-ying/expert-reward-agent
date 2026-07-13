# Training Feedback

## Final-policy outcome
score=224.208876, len=437.100000, terminated=19/20, truncated=1/20, reward_errors=0
score_range=[18.631804, 279.104121]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_proximity | 415.061881 | 70.3% | 70.3% | 100.0% |
| descent_safety | -91.659276 | -15.5% | 15.5% | 77.6% |
| stable_landed | 50.435412 | 8.5% | 8.5% | 13.2% |
| fuel_penalty | -18.005000 | -3.0% | 3.0% | 82.4% |
| touchdown_bonus | 15.247017 | 2.6% | 2.6% | 0.7% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
