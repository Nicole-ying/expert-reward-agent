# Training Feedback

## Final-policy outcome
score=-145.243759, len=167.000000, terminated=18/20, truncated=2/20, reward_errors=0
score_range=[-1413.821829, 27.536940]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| upright_pen | -204.573727 | -71.3% | 71.3% | 57.7% |
| forward | 74.913691 | 26.1% | 28.7% | 43.2% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 2/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
