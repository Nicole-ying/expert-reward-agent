# Training Feedback

## Final-policy outcome
score=-12.659497, len=13.250000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-51.653454, -4.515459]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 4.844773 | 54.0% | 54.0% | 77.0% |
| lateral_penalty | 3.787663 | 42.2% | 42.2% | 100.0% |
| upright_penalty | 0.192176 | 2.1% | 2.1% | 100.0% |
| height_penalty | 0.142584 | 1.6% | 1.6% | 15.8% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 1/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
