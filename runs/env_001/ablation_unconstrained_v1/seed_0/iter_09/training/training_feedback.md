# Training Feedback

## Final-policy outcome
score=-98.081455, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-127.271844, -65.987659]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| proximity_reward | 233.714116 | 97.1% | 97.1% | 100.0% |
| stability_penalty | -5.884839 | -2.4% | 2.4% | 100.0% |
| progress_reward | 0.829422 | 0.3% | 0.5% | 100.0% |
| landing_proxy | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
