# Training Feedback

## Final-policy outcome
score=-114.070278, len=68.400000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-141.582367, -96.181510]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing | 4.534475 | 73.4% | 73.4% | 97.6% |
| progress_delta | 1.184495 | 19.2% | 19.2% | 92.0% |
| fuel_penalty | -0.460000 | -7.4% | 7.4% | 3.4% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
