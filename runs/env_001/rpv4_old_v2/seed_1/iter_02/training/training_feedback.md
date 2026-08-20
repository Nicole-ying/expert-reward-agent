# Training Feedback

## Final-policy outcome
score=-104.327660, len=301.400000, terminated=19/20, truncated=1/20, reward_errors=0
score_range=[-180.870995, 31.475726]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| fuel_penalty | -11.462500 | -50.9% | 50.9% | 76.1% |
| progress | 2.206785 | 9.8% | 28.1% | 100.0% |
| contact_bonus | 3.128335 | 13.9% | 13.9% | 0.3% |
| orientation_penalty | -1.587194 | -7.1% | 7.1% | 5.1% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 5/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
