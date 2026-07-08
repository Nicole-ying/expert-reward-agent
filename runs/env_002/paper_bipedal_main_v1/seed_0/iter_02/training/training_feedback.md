# Training Feedback

## Final-policy outcome
score=318.318151, len=1268.850000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[316.709986, 319.743894]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 504.516605 | 81.9% | 81.9% | 100.0% |
| stability_penalty | -69.142840 | -11.2% | 11.2% | 100.0% |
| energy_penalty | -42.227721 | -6.9% | 6.9% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
