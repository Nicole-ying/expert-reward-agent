# Training Feedback

## Final-policy outcome
score=222.454991, len=442.700000, terminated=17/20, truncated=3/20, reward_errors=0
score_range=[88.813365, 278.966143]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_quality_reward | 403.485551 | 95.4% | 95.4% | 65.3% |
| progress_reward | 12.381417 | 2.9% | 3.2% | 97.7% |
| stability_penalty | -5.812219 | -1.4% | 1.4% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
