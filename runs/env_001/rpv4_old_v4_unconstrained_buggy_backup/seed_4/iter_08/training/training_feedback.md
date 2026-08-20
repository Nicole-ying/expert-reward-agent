# Training Feedback

## Final-policy outcome
score=193.524728, len=512.150000, terminated=17/20, truncated=3/20, reward_errors=0
score_range=[-58.599127, 265.528521]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_contact | 82.892785 | 66.5% | 66.5% | 28.0% |
| fuel_efficiency | -13.891500 | -11.1% | 11.1% | 90.4% |
| attitude_constraint | -11.565371 | -9.3% | 9.3% | 100.0% |
| goal_proximity | -11.035701 | -8.9% | 8.9% | 100.0% |
| settling_stability | -5.240783 | -4.2% | 4.2% | 98.7% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
