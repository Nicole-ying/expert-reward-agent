# Training Feedback

## Final-policy outcome
score=-104.174407, len=68.550000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-123.799657, -87.108232]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress_reward | 1.119209 | 55.5% | 57.5% | 100.0% |
| stability_penalty | -0.807639 | -40.0% | 40.0% | 100.0% |
| landing_bonus | 0.050000 | 2.5% | 2.5% | 0.1% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
