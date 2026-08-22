# Training Feedback

## Final-policy outcome
score=811.130467, len=620.650000, terminated=57/100, truncated=43/100, reward_errors=0
score_range=[43.821643, 1614.628461]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| official_reward | 811.130467 | 98.3% | 100.0% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10099
- early_terminal (<150 steps and score<-50): 0/100
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
