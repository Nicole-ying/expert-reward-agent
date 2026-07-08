# Training Feedback

## Final-policy outcome
score=170.997957, len=526.550000, terminated=17/20, truncated=3/20, reward_errors=0
score_range=[-48.263630, 259.917060]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_proxy | 212.397798 | 98.6% | 98.6% | 99.9% |
| stability_penalty | -1.680744 | -0.8% | 0.8% | 100.0% |
| progress_reward | 1.206307 | 0.6% | 0.6% | 99.1% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
