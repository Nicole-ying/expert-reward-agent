# Training Feedback

## Final-policy outcome
score=212.944108, len=511.900000, terminated=16/20, truncated=4/20, reward_errors=0
score_range=[69.659258, 274.047242]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_success_reward | 37.250000 | 90.7% | 90.7% | 7.3% |
| goal_proximity | 2.638238 | 6.4% | 6.8% | 97.6% |
| living_penalty | -1.023800 | -2.5% | 2.5% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
