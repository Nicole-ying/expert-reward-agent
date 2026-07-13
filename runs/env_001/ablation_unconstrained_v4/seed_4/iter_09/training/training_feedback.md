# Training Feedback

## Final-policy outcome
score=-115.860325, len=68.400000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-139.537566, -92.306814]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_reward | 13.683082 | 42.8% | 42.8% | 0.5% |
| progress | 11.167853 | 35.0% | 36.2% | 100.0% |
| distance_penalty | -6.660297 | -20.9% | 20.9% | 100.0% |
| fuel_penalty | -0.040000 | -0.1% | 0.1% | 0.3% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
