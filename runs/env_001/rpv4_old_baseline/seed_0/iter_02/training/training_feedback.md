# Training Feedback

## Final-policy outcome
score=-98.901539, len=876.950000, terminated=3/20, truncated=17/20, reward_errors=0
score_range=[-170.502936, 290.659645]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_reward | 450.000000 | 99.4% | 99.4% | 0.3% |
| progress_reward | 0.452978 | 0.1% | 0.4% | 100.0% |
| action_penalty | -0.874150 | -0.2% | 0.2% | 99.7% |
| stability_penalty | -0.090665 | -0.0% | 0.0% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
