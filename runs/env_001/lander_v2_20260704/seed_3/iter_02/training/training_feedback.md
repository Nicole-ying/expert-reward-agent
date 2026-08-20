# Training Feedback

## Final-policy outcome
score=149.444491, len=564.950000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-93.757388, 213.548917]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| soft_landing_proxy | 71.853044 | 97.4% | 97.4% | 10.9% |
| approach_reward | 1.073917 | 1.5% | 1.8% | 98.8% |
| stability_penalty | -0.568172 | -0.8% | 0.8% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
