# Training Feedback

## Final-policy outcome
score=25.371183, len=214.400000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[9.280969, 70.851963]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| stability_penalty | -1.803193 | -40.7% | 40.7% | 100.0% |
| progress_reward | 1.293367 | 29.2% | 30.1% | 100.0% |
| action_penalty | -1.290000 | -29.1% | 29.1% | 60.2% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
