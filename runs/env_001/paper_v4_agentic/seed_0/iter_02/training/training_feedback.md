# Training Feedback

## Final-policy outcome
score=97.698965, len=872.650000, terminated=3/20, truncated=17/20, reward_errors=0
score_range=[-325.840631, 185.591390]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_contact_bonus | 365.465579 | 99.2% | 99.2% | 72.3% |
| progress_reward | 1.295489 | 0.4% | 0.4% | 100.0% |
| landing_safety_penalty | 1.399652 | 0.4% | 0.4% | 100.0% |
| x_boundary_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 1/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
